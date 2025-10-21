import { ChromaClient } from "chromadb";

const client = new ChromaClient({
    path: 'http://localhost:8000'
})

async function main() {
    const responce = await client.createCollection({
        name: 'data-test'
    })
    console.log(responce);
}
main();