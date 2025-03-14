# Rust AI-SDK

A Rust library for creating AI-powered agents with tool usage capabilities.

## The Vision
```
rust-ai-sdk/
├── src/
│   ├── lib.rs              # Main entry point and exports
│   ├── provider.rs         # Module declaration and common provider code
│   ├── provider/           # LLM provider implementations
│   │   ├── xai.rs          # xAI provider implementation
│   │   └── openai.rs       # OpenAI-compatible provider
│   ├── model.rs            # Module declaration and common model code
│   ├── model/              # Language model abstractions
│   │   └── chat.rs         # Chat model interface
│   ├── tools.rs            # Module declaration and common tools code
│   ├── tools/              # Tool definition and execution
│   │   └── builtin.rs      # Built-in tools (e.g., math, web search)
│   ├── agent.rs            # Module declaration and common agent code
│   ├── agent/              # Agentic behavior logic
│   │   └── simple.rs       # Simple agent implementation
│   ├── error.rs            # Custom error types
│   └── config.rs           # Configuration structs (e.g., API keys, settings)
├── tests/                  # Integration and unit tests
│   ├── provider_tests.rs
│   └── agent_tests.rs
├── examples/               # Example usage
│   ├── simple_chat.rs
│   └── tool_agent.rs
├── Cargo.toml              # Project manifest
└── README.md               # Documentation
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
