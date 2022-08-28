// Copyright 2022 TF.Text Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef THIRD_PARTY_TENSORFLOW_TEXT_CORE_KERNELS_Phrase_TOKENIZER_H_
#define THIRD_PARTY_TENSORFLOW_TEXT_CORE_KERNELS_Phrase_TOKENIZER_H_

#include <string>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/random/random.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "tensorflow_text/core/kernels/phrase_tokenizer_model_generated.h"
#include "tensorflow_text/core/kernels/sentencepiece/double_array_trie.h"
#include "tensorflow_text/core/kernels/whitespace_tokenizer.h"
#include "tensorflow_text/core/kernels/wordpiece_tokenizer.h"

namespace tensorflow {
namespace text {

class StringVocab : public WordpieceVocab {
 public:
  explicit StringVocab(const std::vector<std::string>& vocab) : vocab_(vocab) {
    for (int i = 0; i < vocab.size(); ++i) {
      index_map_[vocab_[i]] = i;
    }
  }

  LookupStatus Contains(absl::string_view key, bool* value) const override {
    *value = index_map_.contains(key);
    return LookupStatus();
  }

  absl::optional<int> LookupId(absl::string_view key) const {
    auto it = index_map_.find(key);
    if (it == index_map_.end()) {
      return absl::nullopt;
    } else {
      return it->second;
    }
  }

  // Returns the key of `vocab_id` or empty if `vocab_id` is not valid.
  absl::optional<absl::string_view> LookupWord(int vocab_id) const {
    if (vocab_id >= vocab_.size() || vocab_id < 0) {
      return absl::nullopt;
    }
    return vocab_[vocab_id];
  }

  int Size() const { return index_map_.size(); }

 private:
  std::vector<std::string> vocab_;
  absl::flat_hash_map<absl::string_view, int> index_map_;
};

class PhraseTokenizer {
 public:
  // Creates an instance.
  //
  // Args:
  //  * config_flatbuffer: the pointer to the PhraseTokenizerConfig
  //    flatbuffer, which is not owned by this instance and should be kept alive
  //    through the lifetime of the instance.
  static absl::StatusOr<PhraseTokenizer> Create(const void* config_flatbuffer);

  // Tokenizes a string (or series of character codepoints) by Phrase.
  //
  // Example:
  // input = "Show me the way."
  // output = ["Show me", "the", "way."]
  //
  // The input should be UTF-8 but the tokenization is performed on Unicode
  // codepoints.
  //
  // Args:
  //  * input: The UTF-8 string of an input.
  //  * tokens: The output tokens.
  void Tokenize(const absl::string_view input,
                std::vector<std::string>* result_tokens,
                std::vector<int>* result_token_ids);

  // Detokenizer the input into a single string.
  absl::StatusOr<std::string> Detokenize(
      const absl::Span<const int> input) const;

 private:
  // Detokenizer the input into vector of strings.
  absl::StatusOr<std::vector<std::string>> DetokenizeToTokens(
      const absl::Span<const int> input) const;

  // Find the phrase tokens based on the current phrase.
  void FindPhraseTokens(const std::string& cur_phrase,
                        std::vector<std::string>* phrase_tokens,
                        std::vector<int>* phrase_token_ids);

  // Lookup the phrase in the token string from current index.
  void PhraseLookup(const std::string& token, int cur, bool* in_vocab,
                    int* index, int* length);

  std::unique_ptr<StringVocab> vocab_ = nullptr;
  std::unique_ptr<WhitespaceTokenizerConfig> whitespace_config_ = nullptr;
  const PhraseTokenizerConfig* phrase_config_;
  std::string whitespace_config_str_;
  std::unique_ptr<sentencepiece::DoubleArrayTrie> trie_ = nullptr;
  float prob_;
  absl::BitGen gen_;
};

}  // namespace text
}  // namespace tensorflow

#endif  // THIRD_PARTY_TENSORFLOW_TEXT_CORE_KERNELS_Phrase_TOKENIZER_H_
