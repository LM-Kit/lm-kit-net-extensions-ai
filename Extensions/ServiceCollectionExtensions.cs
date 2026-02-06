using LMKit.Integrations.ExtensionsAI.ChatClient;
using LMKit.Integrations.ExtensionsAI.Embeddings;
using LMKit.Model;
using Microsoft.Extensions.AI;
using Microsoft.Extensions.DependencyInjection;

namespace LMKit.Integrations.ExtensionsAI
{
    /// <summary>
    /// Provides extension methods for registering LMKit services with an <see cref="IServiceCollection"/>.
    /// </summary>
    public static class ServiceCollectionExtensions
    {
        /// <summary>
        /// Adds an LMKit <see cref="IChatClient"/> implementation to the service collection using the specified model.
        /// </summary>
        /// <param name="services">The service collection to register the chat client with.</param>
        /// <param name="model">The LMKit model used for chat completions.</param>
        /// <param name="defaultOptions">Optional default chat options applied to every request unless overridden.</param>
        /// <returns>The service collection for chaining.</returns>
        public static IServiceCollection AddLMKitChatClient(
            this IServiceCollection services,
            LM model,
            ChatOptions? defaultOptions = null)
        {
            return AddLMKitChatClient(services, new LMKitChatClient(model, defaultOptions));
        }

        /// <summary>
        /// Adds a pre-constructed LMKit <see cref="IChatClient"/> to the service collection.
        /// </summary>
        /// <param name="services">The service collection to register the chat client with.</param>
        /// <param name="chatClient">The <see cref="LMKitChatClient"/> instance to register.</param>
        /// <returns>The service collection for chaining.</returns>
        public static IServiceCollection AddLMKitChatClient(
            this IServiceCollection services,
            LMKitChatClient chatClient)
        {
            services.AddSingleton<IChatClient>(chatClient);
            return services;
        }

        /// <summary>
        /// Adds an LMKit <see cref="IEmbeddingGenerator{TInput, TEmbedding}"/> to the service collection using the specified model.
        /// </summary>
        /// <param name="services">The service collection to register the embedding generator with.</param>
        /// <param name="model">The LMKit model used for generating text embeddings.</param>
        /// <returns>The service collection for chaining.</returns>
        public static IServiceCollection AddLMKitEmbeddingGenerator(
            this IServiceCollection services,
            LM model)
        {
            return AddLMKitEmbeddingGenerator(services, new LMKitEmbeddingGenerator(model));
        }

        /// <summary>
        /// Adds a pre-constructed LMKit <see cref="IEmbeddingGenerator{TInput, TEmbedding}"/> to the service collection.
        /// </summary>
        /// <param name="services">The service collection to register the embedding generator with.</param>
        /// <param name="embeddingGenerator">The <see cref="LMKitEmbeddingGenerator"/> instance to register.</param>
        /// <returns>The service collection for chaining.</returns>
        public static IServiceCollection AddLMKitEmbeddingGenerator(
            this IServiceCollection services,
            LMKitEmbeddingGenerator embeddingGenerator)
        {
            services.AddSingleton<IEmbeddingGenerator<string, Embedding<float>>>(embeddingGenerator);
            return services;
        }
    }
}
