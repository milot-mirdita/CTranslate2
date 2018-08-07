#pragma once

#include <string>
#include <vector>

#include "models/model.h"
#include "translation_result.h"

namespace ctranslate2 {

  struct TranslationOptions {
    size_t beam_size = 2;
    size_t num_hypotheses = 1;
    size_t max_decoding_steps = 250;
    float length_penalty = 0;
    bool use_vmap = false;
  };

  // This class holds all information required to translate from a model. Copying
  // a Translator instance does not duplicate the model data and the copy can
  // be safely executed in parallel.
  class Translator {
  public:
    Translator(const std::shared_ptr<models::Model>& model);
    Translator(const Translator& other);

    TranslationResult
    translate(const std::vector<std::string>& tokens,
              const TranslationOptions& options);

    std::vector<TranslationResult>
    translate_batch(const std::vector<std::vector<std::string>>& tokens,
                    const TranslationOptions& options);

  private:
    const std::shared_ptr<models::Model> _model;
    std::unique_ptr<Encoder> _encoder;
    std::unique_ptr<Decoder> _decoder;
  };

}