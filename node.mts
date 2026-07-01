import { ChatOllama } from "@langchain/ollama";
import {
  StateGraph,
  START,
  END,
  MessagesAnnotation,
} from "@langchain/langgraph";
import { tool } from "@langchain/core/tools";
import { ToolNode } from "@langchain/langgraph/prebuilt";
import { z } from "zod";
import { HumanMessage } from "@langchain/core/messages";

const fetchWeather = tool(
  async ({ city }: { city: string }) => {
    return {
      city,
      temperature: 25,
      condition: "sunny",
    };
  },
  {
    name: "fetchWeather",
    description: "Fetches the current weather for a given city",
    schema: z.object({
      city: z.string().describe("The city name, for example Cairo"),
    }),
  }
);

const tools = [fetchWeather];

const model = new ChatOllama({
  model: "qwen3:8b",
}).bindTools(tools);

const toolNode = new ToolNode(tools);

const agentNode = async (state: typeof MessagesAnnotation.State) => {
  const response = await model.invoke(state.messages);

  return {
    messages: [response],
  };
};

const shouldContinue = (state: typeof MessagesAnnotation.State) => {
  const lastMessage = state.messages[state.messages.length - 1];

  if ("tool_calls" in lastMessage && lastMessage.tool_calls?.length) {
    return "tool";
  }

  return END;
};

const graph = new StateGraph(MessagesAnnotation)
  .addNode("agent", agentNode)
  .addNode("tool", toolNode)
  .addEdge(START, "agent")
  .addConditionalEdges("agent", shouldContinue)
  .addEdge("tool", "agent")
  .compile();

const response = await graph.invoke({
  messages: [new HumanMessage("What is the weather in Cairo?")],
});

console.log(response.messages.at(-1)?.content);