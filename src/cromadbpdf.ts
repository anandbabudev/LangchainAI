import { ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings } from "@langchain/google-genai";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { RecursiveCharacterTextSplitter } from 'langchain/text_splitter'
import { PDFLoader } from "@langchain/community/document_loaders/fs/pdf";
import { Chroma } from "@langchain/community/vectorstores/chroma";
const model = new ChatGoogleGenerativeAI({
    model: "gemini-2.0-flash",
    temperature: 0.7,
});
const embeddings = new GoogleGenerativeAIEmbeddings({
    model: "embedding-001"
});
const questions = "what is SmartHome Hub";

async function main() {

    const pdfloader = new PDFLoader("book.pdf", {
        splitPages: false
    });

    const docs = await pdfloader.load();
    const splitPages = new RecursiveCharacterTextSplitter({
        separators: [`.\n`]
    });
    const splitterdoc = await splitPages.splitDocuments(docs);

    // ✅ Flatten metadata
    const cleanDocs = splitterdoc.map(doc => ({
        pageContent: doc.pageContent,
        metadata: {
            source: doc.metadata.source,
            page: doc.metadata.page
        }
    }));

    // ✅ Store data
    const vectorStore = await Chroma.fromDocuments(cleanDocs, embeddings, {
        collectionName: "book",
        url: "http://localhost:8000" // no trailing slash
    });
    //create data retival
    const retriever = vectorStore.asRetriever({
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