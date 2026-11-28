// Writes llama.cpp's reference for bench/lfm3-check.html: the prompt's tokens, the logits at every prompt position
// and a greedy continuation, as [n][tokens n][vocab][logits n×vocab][n_gen][gen n_gen] (little-endian int32/f32).
//
// Build against a llama.cpp checkout (CPU only is fine):
//   c++ -O2 -std=c++17 tools/llama-dump.cpp -I$LLAMA/include -I$LLAMA/ggml/include \
//     -L$LLAMA/build/bin -lllama -lggml -lggml-base -Wl,-rpath,$LLAMA/build/bin -o llama-dump
// Run:  ./llama-dump model.gguf "The capital of France is" out.bin 12
#include "llama.h"
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

static int argmax(const float * l, int n) { int b = 0; for (int i = 1; i < n; i++) if (l[i] > l[b]) b = i; return b; }

int main(int argc, char ** argv) {
    if (argc < 5) { fprintf(stderr, "usage: %s model.gguf prompt out.bin n_gen\n", argv[0]); return 1; }
    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    llama_model * model = llama_model_load_from_file(argv[1], mp);
    if (!model) { fprintf(stderr, "could not load %s\n", argv[1]); return 1; }
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const std::string prompt = argv[2];
    // add_special=false: the prompt is exactly what the engine under test reads (add BOS in the text if wanted).
    int n = -llama_tokenize(vocab, prompt.c_str(), prompt.size(), nullptr, 0, false, true);
    std::vector<llama_token> toks(n);
    llama_tokenize(vocab, prompt.c_str(), prompt.size(), toks.data(), n, false, true);
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = 1024; cp.n_batch = 1024; cp.n_ubatch = 1024; cp.n_threads = 4;
    llama_context * ctx = llama_init_from_model(model, cp);
    llama_batch b = llama_batch_init(n, 0, 1);
    for (int i = 0; i < n; i++) { b.token[i] = toks[i]; b.pos[i] = i; b.n_seq_id[i] = 1; b.seq_id[i][0] = 0; b.logits[i] = 1; }
    b.n_tokens = n;
    if (llama_decode(ctx, b)) { fprintf(stderr, "decode failed\n"); return 1; }
    const int nv = llama_vocab_n_tokens(vocab);
    FILE * f = fopen(argv[3], "wb");
    fwrite(&n, 4, 1, f); fwrite(toks.data(), 4, n, f); fwrite(&nv, 4, 1, f);
    for (int i = 0; i < n; i++) fwrite(llama_get_logits_ith(ctx, i), 4, nv, f);
    const int ng = atoi(argv[4]);
    std::vector<int> gen;
    int next = argmax(llama_get_logits_ith(ctx, n - 1), nv);
    llama_batch one = llama_batch_init(1, 0, 1);
    for (int g = 0; g < ng; g++) {
        gen.push_back(next);
        one.token[0] = next; one.pos[0] = n + g; one.n_seq_id[0] = 1; one.seq_id[0][0] = 0; one.logits[0] = 1; one.n_tokens = 1;
        if (llama_decode(ctx, one)) { fprintf(stderr, "decode failed at %d\n", g); return 1; }
        next = argmax(llama_get_logits_ith(ctx, 0), nv);
    }
    fwrite(&ng, 4, 1, f); fwrite(gen.data(), 4, ng, f);
    fclose(f);
    printf("tokens:"); for (int t : toks) printf(" %d", t); printf("\ngreedy:"); for (int t : gen) printf(" %d", t); printf("\n");
    llama_batch_free(b); llama_batch_free(one); llama_free(ctx); llama_model_free(model);
    return 0;
}
