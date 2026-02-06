using LMKit.Agents.Tools;
using LMKit.Model;
using LMKit.TextGeneration;
using LMKit.TextGeneration.Chat;
using LMKit.TextGeneration.Sampling;
using Microsoft.Extensions.AI;
using System.Collections.Concurrent;
using System.Runtime.CompilerServices;
using System.Text.Json;

namespace LMKit.Integrations.ExtensionsAI.ChatClient
{
    /// <summary>
    /// Implements <see cref="IChatClient"/> from Microsoft.Extensions.AI using LMKit's local inference engine.
    /// Supports non-streaming and streaming chat completions, tool/function calling, and configurable
    /// sampling parameters.
    /// </summary>
    public sealed class LMKitChatClient : IChatClient
    {
        private readonly LM _model;
        private readonly ChatOptions? _defaultOptions;

        /// <inheritdoc/>
        public ChatClientMetadata Metadata { get; }

        /// <summary>
        /// Initializes a new instance of the <see cref="LMKitChatClient"/> class.
        /// </summary>
        /// <param name="model">The LMKit model used for chat completions.</param>
        /// <param name="defaultOptions">Optional default chat options applied to every request unless overridden.</param>
        public LMKitChatClient(LM model, ChatOptions? defaultOptions = null)
        {
            _model = model ?? throw new ArgumentNullException(nameof(model));
            _defaultOptions = defaultOptions;
            Metadata = new ChatClientMetadata("LMKit", defaultModelId: model.Name);
        }

        /// <inheritdoc/>
        public async Task<ChatResponse> GetResponseAsync(
            IEnumerable<ChatMessage> messages,
            ChatOptions? options = null,
            CancellationToken cancellationToken = default)
        {
            var effectiveOptions = MergeOptions(options);
            var lmkitHistory = ToLMKitChatHistory(messages);

            using var conversation = new MultiTurnConversation(_model, lmkitHistory);
            ApplyOptions(conversation, effectiveOptions);

            var result = await conversation.RegenerateResponseAsync(cancellationToken).ConfigureAwait(false);

            return BuildChatResponse(result);
        }

        /// <inheritdoc/>
        public async IAsyncEnumerable<ChatResponseUpdate> GetStreamingResponseAsync(
            IEnumerable<ChatMessage> messages,
            ChatOptions? options = null,
            [EnumeratorCancellation] CancellationToken cancellationToken = default)
        {
            var queue = new ConcurrentQueue<ChatResponseUpdate>();
            using var semaphore = new SemaphoreSlim(0);
            bool done = false;
            Exception? backgroundException = null;

            void AfterTextCompletion(object? sender, LMKit.TextGeneration.Events.AfterTextCompletionEventArgs e)
            {
                if (e.SegmentType == TextSegmentType.UserVisible)
                {
                    queue.Enqueue(new ChatResponseUpdate(ChatRole.Assistant, e.Text));
                    semaphore.Release();
                }
            }

            _ = Task.Run(async () =>
            {
                try
                {
                    var effectiveOptions = MergeOptions(options);
                    var lmkitHistory = ToLMKitChatHistory(messages);

                    using var conversation = new MultiTurnConversation(_model, lmkitHistory);
                    ApplyOptions(conversation, effectiveOptions);

                    conversation.AfterTextCompletion += AfterTextCompletion;
                    try
                    {
                        await conversation.RegenerateResponseAsync(cancellationToken).ConfigureAwait(false);
                    }
                    finally
                    {
                        conversation.AfterTextCompletion -= AfterTextCompletion;
                    }
                }
                catch (Exception ex)
                {
                    backgroundException = ex;
                }
                finally
                {
                    done = true;
                    semaphore.Release();
                }
            }, cancellationToken);

            while (!done || !queue.IsEmpty)
            {
                await semaphore.WaitAsync(cancellationToken).ConfigureAwait(false);
                while (queue.TryDequeue(out var chunk))
                {
                    yield return chunk;
                }
            }

            if (backgroundException is not null)
            {
                throw backgroundException;
            }
        }

        /// <inheritdoc/>
        public object? GetService(Type serviceType, object? serviceKey = null)
        {
            if (serviceKey is not null)
            {
                return null;
            }

            if (serviceType == typeof(ChatClientMetadata))
            {
                return Metadata;
            }

            if (serviceType?.IsInstanceOfType(this) == true)
            {
                return this;
            }

            return null;
        }

        /// <inheritdoc/>
        public void Dispose()
        {
            // Model lifecycle is managed externally
        }

        #region Private Helpers

        private ChatOptions? MergeOptions(ChatOptions? requestOptions)
        {
            if (requestOptions is not null)
            {
                return requestOptions;
            }

            return _defaultOptions;
        }

        private LMKit.TextGeneration.Chat.ChatHistory ToLMKitChatHistory(IEnumerable<ChatMessage> messages)
        {
            var history = new LMKit.TextGeneration.Chat.ChatHistory(_model);

            foreach (var message in messages)
            {
                var role = MapRole(message.Role);
                var text = message.Text ?? string.Empty;

                history.AddMessage(role, text);
            }

            return history;
        }

        private static AuthorRole MapRole(ChatRole role)
        {
            if (role == ChatRole.System)
            {
                return AuthorRole.System;
            }

            if (role == ChatRole.User)
            {
                return AuthorRole.User;
            }

            if (role == ChatRole.Assistant)
            {
                return AuthorRole.Assistant;
            }

            if (role == ChatRole.Tool)
            {
                return AuthorRole.Tool;
            }

            throw new NotSupportedException($"Unsupported chat role: {role}");
        }

        private static void ApplyOptions(MultiTurnConversation conversation, ChatOptions? options)
        {
            if (options == null)
            {
                return;
            }

            // Sampling parameters
            if (options.Temperature.HasValue || options.TopP.HasValue || options.TopK.HasValue)
            {
                var sampling = new RandomSampling();

                if (options.Temperature.HasValue)
                {
                    sampling.Temperature = options.Temperature.Value;
                }

                if (options.TopP.HasValue)
                {
                    sampling.TopP = options.TopP.Value;
                }

                if (options.TopK.HasValue)
                {
                    sampling.TopK = options.TopK.Value;
                }

                conversation.SamplingMode = sampling;
            }

            // Max tokens
            if (options.MaxOutputTokens.HasValue)
            {
                conversation.MaximumCompletionTokens = options.MaxOutputTokens.Value;
            }

            // Stop sequences
            if (options.StopSequences is { Count: > 0 })
            {
                foreach (var seq in options.StopSequences)
                {
                    conversation.StopSequences.Add(seq);
                }
            }

            // Repetition penalties
            if (options.FrequencyPenalty.HasValue)
            {
                conversation.RepetitionPenalty.FrequencyPenalty = options.FrequencyPenalty.Value;
            }

            if (options.PresencePenalty.HasValue)
            {
                conversation.RepetitionPenalty.PresencePenalty = options.PresencePenalty.Value;
            }

            // Tools
            bool hasTools = options.Tools is { Count: > 0 };

            if (hasTools)
            {
                foreach (var tool in options.Tools!)
                {
                    if (tool is AIFunction aiFunction)
                    {
                        conversation.Tools.Register(new AIToolAdapter(aiFunction));
                    }
                }

                // Tool mode
                ApplyToolMode(conversation, options.ToolMode);
            }

            // Response format (only if no tools, because Grammar and Tools are mutually exclusive in LMKit)
            if (!hasTools && options.ResponseFormat is ChatResponseFormatJson)
            {
                conversation.Grammar = new Grammar(Grammar.PredefinedGrammar.Json);
            }
        }

        private static void ApplyToolMode(MultiTurnConversation conversation, ChatToolMode? toolMode)
        {
            if (toolMode == null || toolMode == ChatToolMode.Auto)
            {
                conversation.ToolPolicy.Choice = ToolChoice.Auto;
            }
            else if (toolMode == ChatToolMode.None)
            {
                conversation.ToolPolicy.Choice = ToolChoice.None;
            }
            else if (toolMode == ChatToolMode.RequireAny)
            {
                conversation.ToolPolicy.Choice = ToolChoice.Required;
            }
            else if (toolMode is RequiredChatToolMode required && required.RequiredFunctionName is not null)
            {
                conversation.ToolPolicy.Choice = ToolChoice.Specific;
                conversation.ToolPolicy.ForcedToolName = required.RequiredFunctionName;
            }
            else
            {
                conversation.ToolPolicy.Choice = ToolChoice.Auto;
            }
        }

        private ChatResponse BuildChatResponse(TextGenerationResult result)
        {
            var responseMessage = new ChatMessage(ChatRole.Assistant, result.Completion);

            var response = new ChatResponse(responseMessage)
            {
                ModelId = _model.Name,
                FinishReason = MapFinishReason(result.TerminationReason),
                Usage = new UsageDetails
                {
                    InputTokenCount = result.PromptTokenCount,
                    OutputTokenCount = result.GeneratedTokenCount,
                    TotalTokenCount = result.PromptTokenCount + result.GeneratedTokenCount
                }
            };

            return response;
        }

        private static ChatFinishReason? MapFinishReason(TextGenerationResult.StopReason reason)
        {
            return reason switch
            {
                TextGenerationResult.StopReason.EndOfGeneration => ChatFinishReason.Stop,
                TextGenerationResult.StopReason.StopSequenceDetected => ChatFinishReason.Stop,
                TextGenerationResult.StopReason.MaxTokenLimitReached => ChatFinishReason.Length,
                TextGenerationResult.StopReason.ContextSizeLimitExceeded => ChatFinishReason.Length,
                TextGenerationResult.StopReason.ToolInvocationRequested => ChatFinishReason.ToolCalls,
                _ => null
            };
        }

        #endregion

        #region Nested Types

        /// <summary>
        /// Adapts a Microsoft.Extensions.AI <see cref="AIFunction"/> to LMKit's <see cref="ITool"/> interface,
        /// allowing AI functions to be registered in LMKit's tool registry.
        /// </summary>
        private sealed class AIToolAdapter : ITool
        {
            private readonly AIFunction _aiFunction;

            public string Name => _aiFunction.Name;

            public string Description => _aiFunction.Description;

            public string InputSchema
            {
                get
                {
                    var schema = _aiFunction.JsonSchema;
                    return schema.ValueKind != JsonValueKind.Undefined
                        ? schema.GetRawText()
                        : """{"type":"object","properties":{}}""";
                }
            }

            public AIToolAdapter(AIFunction aiFunction)
            {
                _aiFunction = aiFunction ?? throw new ArgumentNullException(nameof(aiFunction));
            }

            public async Task<string> InvokeAsync(string arguments, CancellationToken cancellationToken = default)
            {
                IDictionary<string, object?> args;

                try
                {
                    args = JsonSerializer.Deserialize<Dictionary<string, object?>>(arguments)
                        ?? new Dictionary<string, object?>();
                }
                catch (JsonException)
                {
                    args = new Dictionary<string, object?>();
                }

                var result = await _aiFunction.InvokeAsync(
                    new AIFunctionArguments(args!),
                    cancellationToken).ConfigureAwait(false);

                if (result is null)
                {
                    return "null";
                }

                if (result is string str)
                {
                    return str;
                }

                return JsonSerializer.Serialize(result);
            }
        }

        #endregion
    }
}
