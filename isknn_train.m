 function [PAR] = isknn_train(DATA,HP)
 
% --- Incremental Sparse Kernel NN Prototype-Based Training Function ---
%
%   [PAR] = isknn_train(DATA,HP)
% 
%   Input:
%       DATA.
%           input = input matrix                                [p x N]
%           output = output matrix                              [Nc x N]
%       HP.
%           Dm = Design Method                                  [cte]
%               = 1 -> all data set
%               = 2 -> per class
%           Ss = Sparsification strategy                        [cte]
%               = 1 -> ALD
%           v1 = Sparseness parameter 1                         [cte]
%           Us = Update strategy                                [cte]
%               = 0 -> do not update prototypes
%               = 1 -> wta (lms, unsupervised)
%           eta = Update rate                                   [cte]
%           Ps = Prunning strategy                              [cte]
%               = 0 -> do not remove prototypes
%               = 1 -> score-based method 1 (drift based)
%               = 2 -> score-based method 2 (hits and errors)
%           min_score = score that leads to prune prototype     [cte]
%           max_prot = max number of prototypes ("Budget")      [cte]
%           min_prot = min number of prototypes ("restriction") [cte]
%           Ktype = kernel type ( see kernel_func() )           [cte]
%           sig2n = kernel regularization parameter             [cte]
%           sigma = kernel hyperparameter ( see kernel_func() ) [cte]
%           alpha = kernel hyperparameter ( see kernel_func() ) [cte]
%           theta = kernel hyperparameter ( see kernel_func() ) [cte]
%           gamma = kernel hyperparameter ( see kernel_func() ) [cte]
%   Output:
%       PAR.
%       	Cx = clusters' centroids (prototypes)               [p x Nk]
%           Cy = clusters' labels                               [Nc x Nk]
%           Km = Kernel Matrix of Entire Dictionary             [Nk x Nk]
%           Kmc = Kernel Matrix for each class (cell)           [Nc x 1]
%           Kinv = Inverse Kernel Matrix of Dictionary          [Nk x Nk]
%           Kinvc = Inverse Kernel Matrix for each class (cell) [Nc x 1]
%           score = used for prunning method                    [1 x Nk]
%           class_history = used for prunning method           	[1 x Nk]
%           times_selected = used for prunning method           [1 x Nk]
%           y_h = class prediction                              [Nc x N]

%% SET DEFAULT HYPERPARAMETERS

if ((nargin == 1) || (isempty(HP))),
    PARaux.Dm = 2;          % Design Method
    PARaux.Ss = 1;          % Sparsification strategy
    PARaux.v1 = 0.1;        % Sparseness parameter 1 
    PARaux.Us = 1;          % Update strategy
    PARaux.eta = 0.1;       % Update Rate
    PARaux.Ps = 2;          % Prunning strategy
    PARaux.min_score = -10; % Score that leads to prune prototype
    PARaux.max_prot = Inf;  % Max number of prototypes
    PARaux.min_prot = 1;    % Min number of prototypes
    PARaux.K = 1;           % Number of nearest neighbors (classify)
    PARaux.Ktype = 2;       % Kernel Type (gaussian)
    PARaux.sig2n = 0.001;   % Kernel regularization parameter
    PARaux.sigma = 2;       % Kernel width (gaussian)
    PARaux.alpha = 1;       % Dot product multiplier
    PARaux.theta = 1;       % Dot product add cte 
    PARaux.gamma = 2;       % Polynomial order
	HP = PARaux;
else
    if (~(isfield(HP,'Dm'))),
        HP.Dm = 2;
    end
    if (~(isfield(HP,'Ss'))),
        HP.Ss = 1;
    end
    if (~(isfield(HP,'v1'))),
        HP.v1 = 0.1;
    end
    if (~(isfield(HP,'Us'))),
        HP.Us = 1;
    end
    if (~(isfield(HP,'eta'))),
        HP.eta = 0.1;
    end
    if (~(isfield(HP,'Ps'))),
        HP.Ps = 2;
    end
    if (~(isfield(HP,'min_score'))),
        HP.min_score = -10;
    end
    if (~(isfield(HP,'max_prot'))),
        HP.max_prot = Inf;
    end
    if (~(isfield(HP,'min_prot'))),
        HP.min_prot = 1;
    end
    if (~(isfield(HP,'K'))),
        HP.K = 1;
    end
    if (~(isfield(HP,'Ktype'))),
        HP.Ktype = 2;
    end
    if (~(isfield(HP,'sig2n'))),
        HP.sig2n = 0.001;
    end
    if (~(isfield(HP,'sigma'))),
        HP.sigma = 2;
    end
    if (~(isfield(HP,'alpha'))),
        HP.alpha = 1;
    end
    if (~(isfield(HP,'theta'))),
        HP.theta = 1;
    end
    if (~(isfield(HP,'gamma'))),
        HP.gamma = 2;
    end
end

%% INITIALIZATIONS

% Data Initialization

X = DATA.input;         % Input Matrix
Y = DATA.output;        % Output Matrix

% Problem Initialization

[Nc,N] = size(Y);       % Total of classes and samples

% Init Outputs

PAR = HP;

if (~isfield(PAR,'Cx'))
    PAR.Cx = [];
    PAR.Cy = [];
    PAR.Km = [];
    PAR.Kmc = [];
    PAR.Kinv = [];
    PAR.Kinvc = [];
    PAR.score = [];
    PAR.class_history = [];
    PAR.times_selected = [];
end

yh = -1*ones(Nc,N);

%% ALGORITHM

% Update Dictionary

for n = 1:N,
    
	% Get sample
    DATAn.input = X(:,n);
    DATAn.output = Y(:,n);

    % Get dictionary size (cardinality, number of prototypes)
    [~,mt1] = size(PAR.Cx);
    
    % Init Dictionary (if it is the first sample)
    if (mt1 == 0),
        % Make a guess (yh = 1 => first class)
        yh(1,n) = 1;
        % Add sample to dictionary
        PAR = isknn_dict_grow(DATAn,PAR);
        % Update number of times this prototype has been selected
        PAR.times_selected = 1;
        continue;
    end
    
    % Predict Output
    OUTn = isknn_classify(DATAn,PAR);
    yh(:,n) = OUTn.y_h;
    
    % Update number of times a prototype has been selected
    win = OUTn.win;
    PAR.times_selected(win) = PAR.times_selected(win) + 1;
    
    % Growing Strategy
	PAR = isknn_dict_grow(DATAn,PAR);

	% Get dictionary size (cardinality, number of prototypes)
    [~,mt2] = size(PAR.Cx);
    
    % Update Strategy (if prototype was not added)
    if(mt2-mt1 == 0),
        PAR = isknn_dict_updt(DATAn,PAR);
    else
        % For debug. Display dictionary size when it grows.
        % display(mt2);
    end
    
    % Prunning Strategy (heuristic based)
    PAR = isknn_score_updt(DATAn,OUTn,PAR);
    PAR = isknn_dict_prun(PAR);
    
end

%% FILL OUTPUT STRUCTURE

PAR.y_h = yh;

%% END