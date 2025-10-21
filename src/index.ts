
import { ChatGoogleGenerativeAI } from "@langchain/google-genai";
import { HumanMessage, SystemMessage } from "@langchain/core/messages";

const model = new ChatGoogleGenerativeAI({
    model: "gemini-2.0-flash",
    temperature: 0.7,
});

async function main() {
    const messages = [
        new SystemMessage("Translate the following from English into tamil"),
        new HumanMessage("good morning"),
    ];

    console.log(await model.invoke(messages));
    // console.log(await model.batch([messages]));
}
main();