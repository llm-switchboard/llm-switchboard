# Security Policy

## Reporting a Vulnerability

If you discover a security vulnerability, please report it responsibly by opening a private issue or contacting the maintainers directly. Do not open a public issue for security vulnerabilities.

## Scope

llm-switchboard handles:
- **API keys** (`OPENWEBUI_API_KEY`) — passed via environment variable, never written to disk
- **Local config files** — stored in `~/.config/llm-switchboard/` with `umask 077`
- **Network requests** — to your Open WebUI instance and optionally Google's pricing page

## Security Measures

- Files are created with restrictive permissions (`umask 077`)
- API keys are read from environment variables only, never persisted
- HTTPS warnings are shown when connecting to remote non-HTTPS URLs
- No external dependencies — reduces supply chain risk
