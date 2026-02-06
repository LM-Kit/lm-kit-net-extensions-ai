# LM-Kit.NET Microsoft.Extensions.AI Integration

This package provides [Microsoft.Extensions.AI](https://learn.microsoft.com/en-us/dotnet/ai/microsoft-extensions-ai) integration for [LM-Kit.NET](https://docs.lm-kit.com/lm-kit-net/), enabling local LLM inference through the standard `IChatClient` and `IEmbeddingGenerator` abstractions.

## Installation

```
dotnet add package LM-Kit.NET.Integrations.ExtensionsAI
```

## Quick Start

### Chat Completion

```csharp
using LMKit.Model;
using LMKit.Integrations.ExtensionsAI.ChatClient;
using Microsoft.Extensions.AI;

// Load a model
using var model = LM.LoadFromModelID("gemma3:4b");

// Create an IChatClient
IChatClient client = new LMKitChatClient(model);

// Use it
var response = await client.GetResponseAsync("What is the capital of France?");
Console.WriteLine(response.Text);
```

### Streaming

```csharp
await foreach (var update in client.GetStreamingResponseAsync("Tell me a story"))
{
    Console.Write(update.Text);
}
```

### Embeddings

```csharp
using LMKit.Integrations.ExtensionsAI.Embeddings;

using var embeddingModel = LM.LoadFromModelID("embeddinggemma-300m");
IEmbeddingGenerator<string, Embedding<float>> generator = new LMKitEmbeddingGenerator(embeddingModel);

var embeddings = await generator.GenerateAsync(["Hello world", "Goodbye world"]);
```

### Dependency Injection

```csharp
using LMKit.Integrations.ExtensionsAI;

builder.Services.AddLMKitChatClient(model);
builder.Services.AddLMKitEmbeddingGenerator(embeddingModel);
```

### Tool Calling

```csharp
var options = new ChatOptions
{
    Tools = [AIFunctionFactory.Create((string city) => $"Sunny, 22C in {city}", "get_weather", "Get weather for a city")]
};

var response = await client.GetResponseAsync("What is the weather in Paris?", options);
```

## Supported Features

| Feature | Support |
|---------|---------|
| Chat completion | Yes |
| Streaming | Yes |
| Embeddings | Yes |
| Tool/function calling | Yes |
| Temperature, TopP, TopK | Yes |
| Max output tokens | Yes |
| Stop sequences | Yes |
| Frequency/presence penalty | Yes |
| JSON response format | Yes |
| Token usage reporting | Yes |
| Dependency injection | Yes |

## License

Apache 2.0
