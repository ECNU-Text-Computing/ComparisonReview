import os
from paper import get_papers, get_other_papers
from tqdm import tqdm
import torch
import torch.nn.functional as F
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel
from typing import Any, Tuple
import pandas as pd
import faiss


GENERATE_TRAIN_DATA = True

CHUNK_SPLIT_SYMBOL = "-----CHUNK-----"

SEARCH_TOP_K = 50
RERANK_TOP_K = 25


class Qwen3EmbeddingModel:
    def __init__(
            self,
            model_name: str,
            use_fp16: bool = True,
            devices: str = "cuda"
    ):
        self.devices = devices
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left')
        self.model = AutoModel.from_pretrained(model_name)
        if use_fp16:
            self.model = self.model.half()
        self.model.eval()
        self.model = self.model.to(devices)

    @staticmethod
    def _last_token_pooling(
            last_hidden_states: torch.Tensor,
            attention_mask: torch.Tensor
    ) -> torch.Tensor:
        left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
        if left_padding:
            return last_hidden_states[:, -1]
        else:
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_size = last_hidden_states.shape[0]
            return last_hidden_states[
                torch.arange(batch_size, device=last_hidden_states.device),
                sequence_lengths
            ]

    def encode(
            self,
            text: str,
            task_description: str = 'Given a title and abstract of a paper, retrieve the most similar papers.',
            return_dense: bool = True
    ) -> dict[str, np.ndarray]:
                       

                  
        input_text = f'Instruct: {task_description}\nQuery: {text}'
                
        eod_id = self.tokenizer.convert_tokens_to_ids("<|endoftext|>")
        max_length = 2048
            
        batch_dict = self.tokenizer(
            input_text,
            padding=True,
            truncation=True,
            max_length=max_length - 2,
            return_tensors="pt"
        )
                     
        input_ids = batch_dict["input_ids"]
        attention_mask = batch_dict["attention_mask"]
        input_ids = torch.cat([input_ids, torch.tensor([[eod_id]])], dim=1)
        attention_mask = torch.cat([attention_mask, torch.tensor([[1]])], dim=1)

                 
        batch_dict = {
            "input_ids": input_ids.to(self.devices),
            "attention_mask": attention_mask.to(self.devices)
        }
              
        with torch.no_grad():
            outputs = self.model(**batch_dict)
                          
        embeddings = self._last_token_pooling(outputs.last_hidden_state, batch_dict['attention_mask'])
             
        embeddings = F.normalize(embeddings, p=2, dim=1)
              
        return {'dense_vecs': embeddings.cpu().numpy()[0]}

    @staticmethod
    def search(
            query: str,
            chunks: list[str],
            model: Any,
            index: faiss.IndexFlatIP,
            top_k: int,
    ) -> list[str]:
        query_embedding = model.encode(query, return_dense=True)
        query_dense_embedding = query_embedding['dense_vecs']
        query_dense_embedding = np.array([query_dense_embedding]).astype('float32')
        scores, indices = index.search(query_dense_embedding, top_k)
        ranked_indices = indices[0][np.argsort(scores)[::-1][:top_k]]
        return [chunks[i] for i in ranked_indices.flatten().tolist()]


class Qwen3Reranker:
    def __init__(
            self,
            model_name: str,
            use_fp16: bool = True,
            devices: str = "cuda"
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left')
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        if use_fp16:
            self.model = self.model.half()
        self.model.eval()
        self.model = self.model.to(devices)
                    
        self.token_false_id = self.tokenizer.convert_tokens_to_ids("no")
        self.token_true_id = self.tokenizer.convert_tokens_to_ids("yes")
        self.max_length = 2048
                         
        self.prefix = "<|im_start|>system\nJudge whether the Answer meets the requirements based on the Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n"
        self.suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        self.prefix_tokens = self.tokenizer.encode(self.prefix, add_special_tokens=False)
        self.suffix_tokens = self.tokenizer.encode(self.suffix, add_special_tokens=False)

    @staticmethod
    def _format_instruction(
            instruction: str,
            query: str,
            doc: str
    ) -> str:
        if instruction is None:
            instruction = 'Given a title and abstract of a paper, retrieve the most similar papers.'
        return f"<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {doc}"

    def _process_inputs(
            self,
            pairs: list[str]
    ) -> dict:
        inputs = self.tokenizer(
            pairs, padding=False, truncation='longest_first',
            return_attention_mask=False,
            max_length=self.max_length - len(self.prefix_tokens) - len(self.suffix_tokens)
        )
                                
        for i, ele in enumerate(inputs['input_ids']):
            inputs['input_ids'][i] = self.prefix_tokens + ele + self.suffix_tokens
                   
        inputs = self.tokenizer.pad(inputs, padding=True, return_tensors="pt", max_length=self.max_length)
                
        if torch.cuda.is_available():
            for key in inputs:
                inputs[key] = inputs[key].to(self.model.device)
        return inputs

    @torch.no_grad()
    def compute_score(
            self,
            pairs: list[Tuple[str, str]],
            task: str = None,
            batch_size: int = 4
    ) -> list[float]:
                    
        scores = []
        for i in range(0, len(pairs), batch_size):
            batch = pairs[i:i + batch_size]
                   
            inputs = self._process_inputs([self._format_instruction(task, q, d) for q, d in batch])
            logits = self.model(**inputs).logits[:, -1, :]
            probs = torch.softmax(logits[:, [self.token_false_id, self.token_true_id]], dim=-1)
            scores.extend(probs[:, 1].tolist())
        return scores

    def rerank(
            self,
            query: str,
            chunks: list[str],
            top_k: int,
    ) -> list[str]:
        pairs = [(query, chunk) for chunk in chunks]
        scores = self.compute_score(pairs)
                                                  
                       
                          
        sorted_chunks = [chunk for _, chunk in sorted(zip(scores, chunks), reverse=True)]
        return sorted_chunks[:top_k]


def save_faiss(
        index: faiss.IndexFlatIP,
        chunks: list[str],
        index_path: str,
        chunk_path: str
) -> None:
    faiss.write_index(index, index_path)
    with open(chunk_path, "w") as f:
        f.write(CHUNK_SPLIT_SYMBOL.join(chunks))

def check_is_connected(save_path: str):
                  
    df = pd.read_csv(save_path)
    nodes = set()
    edges = []
    for index, row in df.iterrows():
        node1 = row['paper_id_1']
        node2 = row['paper_id_2']
        nodes.add(node1)
        nodes.add(node2)
        edges.append((node1, node2))

    class UnionFind:
        def __init__(self, elements):
            self.parent = {e: e for e in elements}
            self.rank = {e: 0 for e in elements}

        def find(self, x):
            if self.parent[x] != x:
                self.parent[x] = self.find(self.parent[x])
            return self.parent[x]

        def union(self, x, y):
            root_x = self.find(x)
            root_y = self.find(y)
            if root_x != root_y:
                if self.rank[root_x] > self.rank[root_y]:
                    self.parent[root_y] = root_x
                elif self.rank[root_x] < self.rank[root_y]:
                    self.parent[root_x] = root_y
                else:
                    self.parent[root_y] = root_x
                    self.rank[root_x] += 1

    uf = UnionFind(nodes)
    for u, v in edges:
        uf.union(u, v)
    roots = {uf.find(node) for node in nodes}
    connected = len(roots) == 1
    print(f"Graph is connected: {connected}")


def similarity(
        venue: str,
        split: str = "test",
):
            
    dataset_path = f"../data/{venue}/iclr"
    index_path = f"../data/{venue}/embedding/index.faiss"
    chunk_path = f"../data/{venue}/embedding/chunks.txt"
    save_path = f"../data/{venue}/embedding/similarity.csv" if split == "test" else f"../data/{venue}/embedding/similarity_{split}.csv"
    embedding_path = f"../model/qwen3-embedding-0.6b"
    reranker_path = f"../model/qwen3-reranker-0.6b"

            
    save_dir = os.path.dirname(save_path)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

              
    search_top_k = SEARCH_TOP_K + 1
    rerank_top_k = RERANK_TOP_K + 1

            
    if venue == "iclr_2025":
        dataset_path = f"../data/{venue}/iclr"
        papers = get_papers(dataset_path, split)
    else:
        other_data_path = f"../data/{venue}/submissions/submissions.csv"
        papers = get_other_papers(other_data_path)

           
    embedding_model = Qwen3EmbeddingModel(
        embedding_path,
        devices="cuda:1",
        use_fp16=False,
    )
    reranker = Qwen3Reranker(
        reranker_path,
        devices="cuda:0",
        use_fp16=False,
    )

             
    def create_chunk(title, abstract):
        return f"Title: {title}\nAbstract: {abstract}\n"
    chunks = []
    for i, paper in enumerate(papers):
        chunk = create_chunk(paper.title, paper.abstract)
        chunks.append(chunk)

                      
    dense_embeddings = []
    for chunk in tqdm(chunks, desc="Encoding chunks"):
        result = embedding_model.encode(chunk)
        dense_embeddings.append(result['dense_vecs'])
    dense_embeddings = np.array(dense_embeddings).astype('float32')

                    
    index = faiss.IndexFlatIP(dense_embeddings.shape[1])
    index.add(dense_embeddings)
    save_faiss(index, chunks, index_path, chunk_path)

                   
    graph = np.full((len(papers), len(papers)), 0, dtype=int)
    for i, paper in enumerate(tqdm(papers, desc="Creating paper pairs")):
        query_chunk = chunks[i]
        similar_chunks = Qwen3EmbeddingModel.search(
            query=query_chunk,
            chunks=chunks,
            model=embedding_model,
            index=index,
            top_k=search_top_k
        )
        reranked_chunks = reranker.rerank(
            query=query_chunk,
            chunks=similar_chunks,
            top_k=rerank_top_k
        )
            
        max_cnt = len(reranked_chunks)
        cnt = 0
        for chunk in reranked_chunks:
            j = chunks.index(chunk)
            if i == j:
                continue
            cnt = cnt + 1
            graph[i, j] += max_cnt - cnt
            graph[j, i] = graph[i, j]
        
    indices = np.where(graph > 0)
    sorted_indices = np.argsort(-graph[indices])
    sorted_pairs = [(indices[0][idx], indices[1][idx]) for idx in sorted_indices]
    unique_pairs = [(i, j) for i, j in sorted_pairs if i < j]
    print(f"Total pairs: {len(unique_pairs)}")
              
    id_count = {}
    for i, j in unique_pairs:
        id_count[i] = id_count.get(i, 0) + 1
        id_count[j] = id_count.get(j, 0) + 1
    print("Paper ID occurrence counts:")
    for paper_id in sorted(id_count.keys()):
        print(f"Paper {paper_id}: appears {id_count[paper_id]} time(s)")
        
    df = pd.DataFrame(unique_pairs, columns=['paper_id_1', 'paper_id_2'])
    df.to_csv(save_path, index=False)
              
    check_is_connected(save_path)


if __name__ == "__main__":

    similarity("iclr_2025")
                             
                                                
