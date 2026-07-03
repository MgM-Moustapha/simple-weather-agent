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

const WeatherReportExtraction = z.object({
    cityName: z.string().describe("The city name, for example Cairo"),
    temp: z.number().describe("The temperature in Celsius"),
    conditions: z.string().describe("The weather condition, for example sunny"),
});

const messages: BaseMessage[] = [];

const userInput = [
  "should i pack a heavy jacket for my trip to cairo today?",
  "why does humidity feel different near coastal cities compared to inland deserts?",
]

messages.push(
    new HumanMessage("should i pack a heavy jacket for my trip to cairo today?")
);

const scenarioA = await model.invoke(messages);

messages.push(scenarioA);

if (scenarioA.tool_calls?.length){
    console.log("[system] caught LLM request to run tool fetchWeather");
    
    const toolMessage = await fetchWeather.invoke({cityName: scenarioA.tool_calls[0]?.args.cityName});
    messages.push(new ToolMessage({
        tool_call_id: scenarioA.tool_calls[0]?.id ?? "",
        content: JSON.stringify(toolMessage),
    }));
    const stream = await model.stream(messages);

    for await (const chunk of stream) {
        if (chunk.content && typeof chunk.content === "string"){
            process.stdout.write(chunk.content);
        }
    }
    console.log("");
}

const scenarioB = await model.stream("why does humidity feel different near coastal cities compared to inland deserts?");


for await (const chunk of scenarioB) {
        if (chunk.content && typeof chunk.content === "string"){
            process.stdout.write(chunk.content);
        }
    }
console.log("");

const structuredModel = new ChatOllama({
    model: "qwen3:8b",
}).withStructuredOutput(WeatherReportExtraction);

const scenarioC = await structuredModel.invoke("log the followoing stats into the database: Ryadh is hot at 42C, Tokyo is clear at 21C");

console.log(scenarioC);
