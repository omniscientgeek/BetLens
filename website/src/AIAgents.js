import React from "react";
import ChatPanel from "./ChatPanel";

function AIAgents() {
  return (
    <div className="ai-agents-page">
      <ChatPanel agentMode={true} debugMode={false} />
    </div>
  );
}

export default AIAgents;
