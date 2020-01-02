import copy
import torch
import torch.nn as nn

# not used
def copy_model(model, share_params=True):
    """
    Copies model structure and, optionally, makes references to parameters point to the base model's ones
    """
    other_model = copy.deepcopy(model)
    params = model.named_parameters()
    other_params = other_model.named_parameters()

    dict_other_params = dict(other_params)

    if share_params:
        for name, param in params:
            if name in dict_other_params:
                dict_other_params[name].data = param.data
                dict_other_params[name].grad = param.grad

    return other_model

"""
Multiflow unit.

Takes a nn.modules object and makes n independant copies.
Outputs the sum, for each copy i, of w(i) * output module(i)
with  w(i) = 1 / ( 1+ exp(- pi * gi )),
   - gi is an unbounded trainable parameter
   - pi is defined by the 'plasticity_scheduler'
       - when linear (min, max), it is defined by p: [0, 1] -> [min, max]

Init params: 'random': small random values

weight_state_dict: state dict to initialize weights of each copy
"""
class MultiflowUnit(nn.Module):
    def __init__(self, model, n_domains, n_streams, init_params='random', plasticity_scheduler='linear', min=0.2, max=2, state_dict=None):

        super(MultiflowUnit, self).__init__()

        self.n_domains = n_domains
        self.gate_activation_function = nn.Softmax(dim=1)

        #self.models = [type(model)() for n in range(n_streams)]
        self.models = nn.ModuleList([copy.deepcopy(model) for n in range(n_streams)])

        if state_dict:
            # update state of each copy using 'state_dict' , if provided
            pretrained_dict = {k: v for k, v in state_dict.items() if k in model}
        else:
            # otherwise, copy weights and stuffs from the 'model' object
            pretrained_dict = model.state_dict()

        print("DEBUG model state dict: ", pretrained_dict.keys())
        print("DEBUG self.model state dict: ", self.models[0].state_dict().keys())

        for model in self.models:
            model.load_state_dict(pretrained_dict)

        # initialize each gate in the range [-0.01, 0.01]
        if init_params == 'random':
            expected_size = (n_domains, n_streams)
            initial_value = 0.01 * (torch.rand(*expected_size) * 2 - 1)

        # the trainable params of our multiflow unit :  gate_param[domain][stream]
        self.gate_param = nn.Parameter(initial_value, requires_grad=True)

        # plasticity scheduler: for now only 'linear' is supported
        if plasticity_scheduler == 'linear':
            self.placticity = lambda progress: min + (max - min) * progress


    def forward(self, x, y, param = None):

        assert 'progress' in param
        progress = float(param['progress'])
        p = self.plasticity(progress)

        """
        # TODO verify compute weight for each flow
        #1 stream:
        [[0.2]
         [0.4]]
         # domains : [1, 1, 0, 1]

         # output:  [0.4, 0.4, 0.2, 0.4]

        # 2 streams: 2 domains

        [0.4, 0.3]
        [0.8, 0.7]

        domains: [1,1,0]

        outputs of streams x1, x2

        output:  [0.8 * x1[0] + 0.7 * x2[0],  0.8 * x1 [1] + 0.7 * x2[1], 0.4 * x1[2] + 0.3 * x2(2) ]

       =   (x10  x11 x12 x13 ) *   (0.8  0.8  0.4)
       +  (x20 x21 x22 x23) * (0.7   0.7   0.3)

       weights =  [gate_param[i][j]  for i in domains (j = flow number)]


       or weights =

(x11  x12  x13)    *  (0, 1) *  (0.4, 0.3)  = (x11 x12 x13) * ( 0.8 , 0.7 )           // .T =  (0.8, 0.8, 0.4)  *  (x11  x21) = ()
(x21  x22  x23)       (0, 1)    (0.8, 0.7)    (x21 x22 x23)   ( 0.8 , 0.7 )           //       (0.7, 0.7, 0.3)     (x12  x22)
                      (1, 0)                                  ( 0.4 , 0.3 )           //                           (x13  x23)

        """



        domains = y[3].data
        batch_size = domains.size(0)

        dummies_domain = torch.zeros(batch_size, self.n_domains)
        dummies_domain[:, domains] = 1

        # compute weights for each flow
        gate = self.gate_activation_function(self.gate_param * p)

        print("DEBUG: domains: ", domains, "dummies: ", dummies_domain, "gate param: ", gate)


        weights = (dummies_domain @ gate)

        responses = torch.stack([model(x) for model in self.models])

        # return weighted sum
        result = (responses @ weight).sum(dim=0)
        return result
