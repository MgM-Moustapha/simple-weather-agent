import { ChatOllama } from "@langchain/ollama";
import { tool } from "@langchain/core/tools";
import {
  HumanMessage,
  AIMessage,
  ToolMessage,
  SystemMessage,
  BaseMessage,
} from "@langchain/core/messages";
import { z } from "zod";
import * as readline from 'node:readline/promises'

const rl = readline.createInterface({
    input: process.stdin,
    output: process.stdout
});

const fetchWeather = tool(
  async ({ cityName }: { cityName: string }) => {
    return {
      cityName,
      temp: 25,
      conditions: "sunny",
    };
  },
  {
    name: "fetchWeather",
    description: "Fetches the current weather for a given city",
    schema: z.object({
      cityName: z.string().describe("The city name, for example Cairo"),
    }),
  }
);

const model = new ChatOllama({
  model: "qwen3:8b",
}).bindTools([fetchWeather]);

async function streamMessages(messages: BaseMessage[]) {
  const chunks = await model.stream(messages);
  let fullContent = "";

  for await (const chunk of chunks) {
    if (chunk.content && typeof chunk.content === "string") {
      process.stdout.write(chunk.content);
      fullContent += chunk.content;
    }
  }

  process.stdout.write("\n");
  return fullContent;
}

const pruneHistory = (chatHistory: BaseMessage[], maxMessages: number) => {
    let newArray: BaseMessage[] = []
    const sysMessage = chatHistory[0];
    if(chatHistory.length > maxMessages){
        const index = chatHistory.length - maxMessages
        if(chatHistory[index]?._getType() === "tool"){
            newArray = chatHistory.slice(index - 1)
            newArray.unshift(sysMessage!)
        }else{
            newArray = chatHistory.slice(index)
            newArray.unshift(sysMessage!)
        }
    }else{
        newArray = chatHistory;
    }
    return newArray;
}

let chatHistory: BaseMessage[] = [];

chatHistory.push(
    new SystemMessage("you are a helpful ai agent, you have to return to the previous messages if the user is referring to them or using prounous like 'there'.")
)

async function main() {
  while (true) {
    const answer = await rl.question("How can i assist you?(Q to quit):  ");

    if (answer.toLowerCase() === "q") {
      break;
    }

    chatHistory.push(new HumanMessage(answer));

    const response = await model.invoke(chatHistory);

    chatHistory.push(response);

    if (response.tool_calls?.length) {
      for (const toolCall of response.tool_calls) {
        console.log("[system] caught LLM request to run tool fetchWeather");

        const toolResult = await fetchWeather.invoke({
          cityName: toolCall.args.cityName,
        });

        chatHistory.push(
          new ToolMessage({
            tool_call_id: toolCall.id ?? "",
            content: JSON.stringify(toolResult),
          })
        );
      }

      const finalContent = await streamMessages(chatHistory);

      chatHistory.push(new AIMessage(finalContent));
    } else {
      console.log(response.content)
    }
    chatHistory = pruneHistory(chatHistory, 6)
  }

  rl.close();
}

main().catch((error) => {
    console.log(error);
    process.exit;
})