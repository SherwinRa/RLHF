import torch


def two_valued_feedback_function(a, b) -> torch.Tensor:
    """Predicts the probabilities of preferring options a and b respectively.

    This function is from the paper "Deep Reinforcement Learning from Human Preferences" by Christiano and Leike et al.

    Args:
        a : The reward value of option a.
        b : The reward value of option b.

    Returns:
        torch.Tensor: A tensor containing the probability of preferring option a and the probability of preferring option b.
    """
    a = torch.as_tensor(a)
    b = torch.as_tensor(b)
    
    p = 1 / (1.0 + torch.exp(b-a))
    return torch.stack([p, 1-p])

def three_valued_feedback_function(a, b, eps) -> torch.Tensor:
    """Predicts the probabilities of preferring options a, b and uncertainty respectively.

    This function is based on the feedback function from thepaper "Deep Reinforcement Learning from Human Preferences" 
    by Christiano and Leike et al.

    Args:
        a : The reward value of option a.
        b : The reward value of option b.
        eps : The epsilon value that determines the uncertainty region. If epsilon is less than or equal to -100, then the two-valued feedback function is directly used.

    Returns:
        torch.Tensor: A tensor containing the probability of preferring option a, option b, and being uncertain.
    """
    if(eps <= -100):
        return two_valued_feedback_function(a, b)
    
    a = torch.as_tensor(a)
    b = torch.as_tensor(b)
    eps = torch.as_tensor(eps)

    difference = torch.abs(a - b)
    pq_uncertain = two_valued_feedback_function(2*eps, difference)
    p_uncertain, q_uncertain = pq_uncertain[0], pq_uncertain[1]
    p_a, p_b = two_valued_feedback_function(a, b) * q_uncertain

    return torch.stack([p_a, p_b, p_uncertain])