
## Initial thoughts
_Wed, Mar 25, 2026 at 11:11 PM_

The detect will possible be code to find the stale lines, outline process, potentionsl errors.
Detecting potential errors might be best with AI.
The analyze will probably be be code with a blend of Ai.
The brief will be AI. The challenge will be sending enough information to help AI understand the context.
I am thinking of showing a overview of bets that with icons for the issues. Possible difference of low, medium, high. I might have different agents for each type of discrepancy with unique system prompts. 

I am going to have the frontend as react and back-end as python.

At first I was thinking of having RAG and MCP for calculation. I might use MCP to allow the AI to get better context of the issue while keeping the context window small.

## Current plan
_Thu, Mar 26, 2026 at 08:07 AM_

I am going to calculate the odds for each line, then determine the best line. Then work on the best line. Finally, start on the AI integration. I am leaning towards using Claude with a previous integration I use. This emulates the terminal and allows me to use the Claude Scription without the Claude API penalty cost. I am thinking of testing different AI integrations to see if there are any differences. I will need to research how well different AI models handle sports betting. I have a feeling the frontier models will all perform similarly.

## Websockets
_Thu, Mar 26, 2026 at 09:29 AM_

I am going to use websockets for providing the latest status. This will help the website be responsive and display the latest without the webpage keep requesting for the latest data.
