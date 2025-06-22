from modules.classes.AttentionExtractor import AttentionExtractor

class AttentionAnalyzer:
  def __init__(self, extractor: AttentionExtractor):
    self.extractor = extractor
    self.attentions = [attentions for attentions in self.extractor]
    # extract attention heads and put it in a format that is good for doing statistics and analysis

  def print_attention_shape(self):
    for a in self.attentions:
      print(a.shape)

  
