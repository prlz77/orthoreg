engine.hooks.onBackward = function(state)
  -- OrthoReg
  -- params: 
  --    opt.beta: the regularization stepSize (0.001 is recommended)
  --    opt.learningRate: the current learning rate
  --    opt.lambda: the influence radius of the feature weights. Recommended:10.
  --    opt.epsilon: set to 0.0000001 for numerical stability.
  local modules = state.network:findModules('cudnn.SpatialConvolution')
  for i = 1, #modules do --loop through all conv layers
    local m = modules[i]
    local filters = m.weight:clone():view(m.weight:size(1),-1)
    local norms = filters:norm(2,2):squeeze()
    norms = norms:view(-1,1):expandAs(filters)
    filters:cdiv(norms + opt.epsilon)
    local grad = torch.mm(filters, filters:transpose(2,1))
    grad = torch.exp(grad*opt.lambda)
    grad = (grad * opt.lambda):cdiv(grad + torch.exp(10)) --squashing
    grad[torch.eye(grad:size(1)):byte()] = 0 -- zero out diagonal
    grad = torch.mm(grad, filters)
    local weight = m.weight:view(m.weight:size(1), -1)
    local coef = opt.beta*-1
    coef = coef * opt.learningRate
    weight:add(grad*coef) -- update weights
  end
end

