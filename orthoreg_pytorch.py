    def apply_orthoreg(model, lr, beta=0.001, lambd=10., epsilon=1e-6):
        """Loops through the layers of a CNN and applies orthoreg regularization.
           Apply it before "zero_grad()" or after "step()"
           Rodríguez, Pau, et al. "Regularizing cnns with locally constrained decorrelations." 
           ICLR (2017).
        
        Arguments:
            model {torch.nn.Module} -- network to regularize
            lr {float} -- current model learning rate
        
        Keyword Arguments:
            beta {float} -- regularization strength (default: {0.001})
            lambd {float} -- dampening (default: {10.})
            epsilon {[type]} -- numerical stability constant (default: {1e-6})
        """
        @torch.no_grad()
        def orthoreg(m):
            if type(m) == torch.nn.Conv2d:
                filters = m.weight.data.clone().view(m.weight.size(0), -1)
                norms = filters.norm(2, 1).view(-1, 1).expand_as(filters)
                filters.div_(norms + epsilon)
                grad = torch.mm(filters, filters.transpose(1, 0))
                grad = (grad * lambd) / (grad + np.exp(lambd))
                grad = grad * (1 - torch.eye(grad.size(0), dtype=grad.dtype, device=grad.device))
                grad = torch.mm(grad, filters)
                coeff = -1 * beta * lr
                m.weight.data.view(m.weight.size(0), -1).add_(grad * coeff)
        model.apply(orthoreg)
