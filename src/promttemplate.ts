import { ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings } from "@langchain/google-genai";
import { MemoryVectorStore } from "langchain/vectorstores/memory"
import { Document } from "langchain/document"
import { ChatPromptTemplate } from "@langchain/core/prompts";

const model = new ChatGoogleGenerativeAI({
    model: "gemini-2.0-flash",
    temperature: 0.7,
});

const mydata = [
    "my name is anand",
    "my name is bob",
    "my favourite food is friedrice"
];
const questions = "what is your favourite food";

async function main() {
    //store data
    const vectorstore = new MemoryVectorStore(new GoogleGenerativeAIEmbeddings);
    await vectorstore.addDocuments(mydata.map(
        content => new Document({ pageContent: content })
    ));

    //create data retival
    const retriever = vectorstore.asRetriever({
        k: 2
    });

    // get retrival document
    const results = await retriever.getRelevantDocuments(questions);

    const resultdocs = results.map(
        result => result.pageContent
    );

    const systemTemplate = "Answer the users question based on the following context: {context}";
    //build template
    const promptTemplate = ChatPromptTemplate.fromMessages([
        ["system", systemTemplate],
        ["user", "{input}"],
    ]);

    const chain = promptTemplate.pipe(model);

    const response = await chain.invoke({
        input: questions,
        context: resultdocs
    })
    console.log(response.content);
}
main()