# Model Context Protocol (MCP)

Model Context Protocol is an open-source standard for connecting AI applications to external systems.

Using MCP, AI applications like Claude or ChatGPT can connect to data sources (e.g. local files, databases), tools (e.g. search engines, calculators) and workflows (e.g. specialized prompts) - enabling them to access key information and perform tasks.

Think of MCP like a USB-C port for AI applications. Just as USB-C provides a standardized way to connect electronic devices, MCP provides a standardized way to connect AI applications to external systems.

```Bash
Chat interface <-------------->                         <---> Data and file systems
(Claude Desktop, LibreChat)                                   (PostgreSQL, SQLite, GDrive)

IDEs and code editors <------->          MCP            <---> Development tools
(Claude Code, Goose)            Standardized protocol         (Git, Sentry, etc.)

Other AI applications <------->                         <---> Productivity tools
(5ire, Superinterface)                                        (Slack, Google Maps, etc.)

AI applications                Bidirectional data flow         Data sources and tools
```


---

### Reference
- Model Context Protocol, https://modelcontextprotocol.io/docs/getting-started/intro, 2026-03-12-Thu.
- Model Context Protocol Architecture, https://modelcontextprotocol.io/docs/learn/architecture, 2026-03-12-Thu.
