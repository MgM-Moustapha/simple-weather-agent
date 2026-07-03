import { ChatOllama } from "@langchain/ollama";
import { tool } from "@langchain/core/tools";
import {
  HumanMessage,
  AIMessage,
  ToolMessage,
  BaseMessage,
} from "@langchain/core/messages";
import { z } from "zod";

const fetchWeather = tool(
  async ({ city}: { city: string}) => {
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

const model = new ChatOllama({
  model: "qwen3:8b",
}).bindTools([fetchWeather]);

const WeatherReportExtraction = z.object({
    city: z.string().describe("The city name, for example Cairo"),
    temperature: z.number().describe("The temperature in Celsius"),
    condition: z.string().describe("The weather condition, for example sunny"),
});

const messages: BaseMessage[] = [];

const userInput = [
  "should i pack a heavy jacket for my trip to cairo today?",
  "why does humidity feel different near coastal cities compared to inland deserts?",
]

messages.push(
    new HumanMessage(userInput[1]!)
);

const response = await model.invoke(messages);

messages.push(response);

if (response.tool_calls?.length){
    console.log("[system] caught LLM request to run tool fetchWeather");
    
    const toolMessage = await fetchWeather.invoke({city: response.tool_calls[0]?.args.city});
    messages.push(new ToolMessage({
        tool_call_id: response.tool_calls[0]?.id ?? "",
        content: JSON.stringify(toolMessage),
    }));
    const stream = await model.stream(messages);

    for await (const chunk of stream) {
        if (chunk.content && typeof chunk.content === "string"){
            process.stdout.write(chunk.content);
        }
    }
    console.log("");
}else{
    console.log("[system] no tool calls were made by the LLM");
    const stream = await model.stream(messages);

    for await (const chunk of stream) {
        if (chunk.content && typeof chunk.content === "string"){
            process.stdout.write(chunk.content);
        }
    }
    console.log("");
}

const structuredModel = new ChatOllama({
    model: "qwen3:8b",
}).withStructuredOutput(WeatherReportExtraction);

const logMessage = await structuredModel.invoke("log the followoing stats into the database: Ryadh is hot at 42C, Tokyo is clear at 21C");

console.log(logMessage);
