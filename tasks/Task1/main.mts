import { ChatOllama } from "@langchain/ollama";
import { tool } from "@langchain/core/tools";
import {
  HumanMessage,
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

const WeatherReportsExtraction = z.array(WeatherReportExtraction);

const messages: BaseMessage[] = [];

async function runScenarioA() {

  messages.push(
      new HumanMessage("should i pack a heavy jacket for my trip to cairo today?")
  );

  const scenarioA = await model.invoke(messages);

  messages.push(scenarioA);

  for (const toolCall of scenarioA.tool_calls ?? []) {
      console.log("[system] caught LLM request to run tool fetchWeather");
      
      const toolMessage = await fetchWeather.invoke({cityName: toolCall.args.cityName});
      messages.push(new ToolMessage({
          tool_call_id: toolCall.id ?? "",
          content: JSON.stringify(toolMessage),
      }));
  }
  const stream = await model.stream(messages);

  for await (const chunk of stream) {
      if (chunk.content && typeof chunk.content === "string"){
          process.stdout.write(chunk.content);
      }
  }
  console.log("");
}

async function runScenarioB() {
  const structuredModel = new ChatOllama({
    model: "qwen3:8b",
  }).withStructuredOutput(WeatherReportsExtraction);

  const scenarioB = await structuredModel.invoke("log the followoing stats into the database: Ryadh is hot at 42C, Tokyo is clear at 21C");

  console.log(scenarioB);
}

async function runScenarioC() {

  const scenarioC = await model.stream("why does humidity feel different near coastal cities compared to inland deserts?");


  for await (const chunk of scenarioC) {
          if (chunk.content && typeof chunk.content === "string"){
              process.stdout.write(chunk.content);
          }
      }
  console.log("");
}

async function main() {
  console.log("Running scenario A...");
  await runScenarioA();
  console.log("Running scenario B...");
  await runScenarioB();
  console.log("Running scenario C...");
  await runScenarioC();
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
