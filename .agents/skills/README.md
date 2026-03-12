# Agent Skills

Agent Skills provide modular, on-demand expertise for AI agents. They are discovered via the filesystem and triggered based on the description in their `SKILL.md` metadata. For general information on creating and using skills, visit [AgentSkills.io](https://agentskills.io/).

## Gemini CLI

Place skills in `.agents/skills/` or `~/.gemini/skills/`. For installing from this git repository:

```bash
gemini skills install https://github.com/gdsfactory/gdsfactory.git --path .agents/skills/gdsfactory-component-designer
```

## Claude Code

Place skills in `.agents/skills/` or `~/.claude/skills/`. Claude discovers them automatically at startup.

## Others

Check your agent's documentation for how to add skills, as the process may vary. Alternatively, you may use [vercel-labs/skills](https://github.com/vercel-labs/skills) CLI to manage skills across different agents:

```bash
npx skills add gdsfactory/gdsfactory --skill gdsfactory-component-designer
```
