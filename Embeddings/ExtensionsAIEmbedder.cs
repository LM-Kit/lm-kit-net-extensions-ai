using LMKit.Abstractions;
using Microsoft.Extensions.AI;

namespace LMKit.Integrations.ExtensionsAI.Embeddings
{
    /// <summary>
    /// Adapts a Microsoft.Extensions.AI <see cref="IEmbeddingGenerator{TInput, TEmbedding}"/> to
    /// LMKit's <see cref="IEmbedder"/>, so any embedding provider in the .NET AI ecosystem
    /// (Azure OpenAI, OpenAI, Amazon Bedrock, Ollama, and others) can be used anywhere LMKit
    /// consumes embeddings, including <c>RagEngine</c>.
    /// </summary>
    /// <remarks>
    /// This is the inverse of <see cref="LMKitEmbeddingGenerator"/>, which exposes an LMKit
    /// <see cref="LMKit.Embeddings.Embedder"/> as an <see cref="IEmbeddingGenerator{TInput, TEmbedding}"/>.
    /// The embedding dimension is learned from the first response when not supplied.
    /// </remarks>
    public sealed class ExtensionsAIEmbedder : EmbedderBase
    {
        private readonly IEmbeddingGenerator<string, Embedding<float>> _generator;
        private readonly string _modelId;
        private int _embeddingSize;

        /// <summary>
        /// Initializes a new instance of the <see cref="ExtensionsAIEmbedder"/> class.
        /// </summary>
        /// <param name="generator">The Microsoft.Extensions.AI embedding generator to wrap. Cannot be null.</param>
        /// <param name="modelId">
        /// An optional identifier for the underlying model. When null, the generator's advertised
        /// default model id is used, falling back to a generic label.
        /// </param>
        /// <param name="embeddingSize">
        /// The vector dimension, when known ahead of time. When 0, it is learned from the first
        /// successful embedding call.
        /// </param>
        /// <exception cref="ArgumentNullException">Thrown when <paramref name="generator"/> is null.</exception>
        public ExtensionsAIEmbedder(
            IEmbeddingGenerator<string, Embedding<float>> generator,
            string? modelId = null,
            int embeddingSize = 0)
        {
            _generator = generator ?? throw new ArgumentNullException(nameof(generator));

            var metadata = generator.GetService(typeof(EmbeddingGeneratorMetadata)) as EmbeddingGeneratorMetadata;
            _modelId = modelId ?? metadata?.DefaultModelId ?? "extensions-ai";
            _embeddingSize = embeddingSize;
        }

        /// <inheritdoc/>
        public override string ModelId => _modelId;

        /// <inheritdoc/>
        public override int EmbeddingSize => _embeddingSize;

        /// <inheritdoc/>
        public override async Task<float[][]> GetEmbeddingsAsync(
            IEnumerable<string> texts,
            CancellationToken cancellationToken = default)
        {
            if (texts is null)
            {
                throw new ArgumentNullException(nameof(texts));
            }

            GeneratedEmbeddings<Embedding<float>> generated =
                await _generator.GenerateAsync(texts, options: null, cancellationToken).ConfigureAwait(false);

            var result = new float[generated.Count][];
            for (int i = 0; i < generated.Count; i++)
            {
                result[i] = generated[i].Vector.ToArray();
            }

            if (_embeddingSize == 0 && result.Length > 0)
            {
                _embeddingSize = result[0].Length;
            }

            return result;
        }
    }
}
