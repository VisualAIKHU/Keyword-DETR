import torch
import torch.nn as nn
import torch.nn.functional as F
from .finch import FINCH

class KeywordWeighting(nn.Module):
    def __init__(self, temperature=1.0):
        super(KeywordWeighting, self).__init__()
        self.tau = temperature

    def forward(self, src_vid, src_txt, src_txt_mask):
        """
        Args:
            src_vid: Video features (B, L_vid, D_vid)
            src_txt: Text features (B, L_txt, D_txt)
            src_txt_mask: Text mask (B, L_txt)
        Returns:
            keyword_weight: Normalized keyword weights (B, L_txt)
        """
        with torch.no_grad():
            # FINCH clustering
            clust = []
            max_cluster_num = 0
            
            for i in range(src_vid.shape[0]):
                c, num_clust, _ = FINCH(src_vid[i], initial_rank=None, req_clust=None, 
                                      distance='cosine', ensure_early_exit=True, tw_finch=True)
                
                if len(num_clust) > 1:
                    clust.append(c[:, 1])
                    num_clust = num_clust[1]
                else:
                    clust.append(c[:, 0])
                    num_clust = num_clust[0]
                    
                max_cluster_num = max(max_cluster_num, num_clust)
            
            clust = torch.stack(clust, dim=0)
            
            # Create cluster representations
            one_hot = F.one_hot(clust, num_classes=max_cluster_num)
            cluster_sum = torch.einsum('blc,bld->bcd', one_hot.float(), src_vid)
            cluster_count = one_hot.sum(dim=1)
            
            # Compute cluster means
            cluster_rep = cluster_sum / (cluster_count.unsqueeze(-1) + 1e-8)
            cluster_rep_mask = (cluster_count > 0)
            cluster_rep = torch.where(cluster_rep_mask.unsqueeze(-1), 
                                    cluster_rep, 
                                    torch.zeros_like(cluster_rep))
            
            # Compute attention weights
            dot_product = torch.einsum('bcd,bld->blc', cluster_rep, src_txt)
            empty_cluster_mask = ~cluster_rep_mask.unsqueeze(1)
            masked_dot_product = torch.where(empty_cluster_mask, 
                                           torch.tensor(float('-inf')).to(dot_product.device), 
                                           dot_product)
            
            # Apply temperature-scaled softmax
            softmax_output = F.softmax(masked_dot_product / self.tau, dim=2)
            keyword_weight = torch.max(softmax_output, dim=2)[0]
            
            # Apply text mask and normalize
            keyword_weight = keyword_weight * src_txt_mask
            keyword_weight = (keyword_weight - keyword_weight.min(dim=1, keepdim=True)[0]) / \
                           (keyword_weight.max(dim=1, keepdim=True)[0] - 
                            keyword_weight.min(dim=1, keepdim=True)[0])
            
        return keyword_weight

def main():
    # Test parameters
    batch_size = 2
    vid_len = 10
    txt_len = 5
    feature_dim = 256
    
    # Create dummy data
    src_vid = torch.randn(batch_size, vid_len, feature_dim)
    src_txt = torch.randn(batch_size, txt_len, feature_dim)
    src_txt_mask = torch.ones(batch_size, txt_len)
    
    # Initialize model
    keyword_weighting = KeywordWeighting(temperature=1.0)
    
    # Forward pass
    keyword_weights = keyword_weighting(src_vid, src_txt, src_txt_mask)
    
    # Print results
    print(f"Input shapes:")
    print(f"Video features: {src_vid.shape}")
    print(f"Text features: {src_txt.shape}")
    print(f"Text mask: {src_txt_mask.shape}")
    print(f"\nOutput shape: {keyword_weights.shape}")
    print(f"Weight stats - Min: {keyword_weights.min():.4f}, Max: {keyword_weights.max():.4f}, Mean: {keyword_weights.mean():.4f}")

if __name__ == "__main__":
    main()