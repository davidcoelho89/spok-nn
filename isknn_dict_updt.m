function [PAR] = isknn_dict_updt(DATA,HP)

% --- Procedure for Dictionary Prunning ---
%
%   [PAR] = isknn_dict_updt(DATA,HP)
%
%   Input:
%       DATA.
%           xt = attributes of sample                           [p x 1]
%           yt = class of sample                                [Nc x 1]
%       PAR.
%           Cx = Attributes of input dictionary                 [p x Nk]
%           Cy = Classes of input dictionary                    [Nc x Nk]
%           Km = Kernel matrix of dictionary                    [Nk x Nk]
%           Kmc = Kernel Matrix for each class (cell)           [Nc x 1]
%           Kinv = Inverse Kernel matrix of dicitionary         [Nk x Nk]
%           Kinvc = Inverse Kernel Matrix for each class (cell) [Nc x 1]
%           score = used for prunning method                    [1 x Nk]
%           class_history = used for prunning method           	[1 x Nk]
%           times_selected = used for prunning method           [1 x Nk]
%           Dm = Design Method                                  [cte]
%               = 1 -> all data set
%               = 2 -> per class
%           Us = Update strategy                                [cte]
%               = 0 -> do not update prototypes
%               = 1 -> lms (wta)
%           eta = Update rate                                   [cte]
%           Ktype = kernel type ( see kernel_func() )           [cte]
%           sig2n = kernel regularization parameter             [cte]
%           sigma = kernel hyperparameter ( see kernel_func() ) [cte]
%           alpha = kernel hyperparameter ( see kernel_func() ) [cte]
%           theta = kernel hyperparameter ( see kernel_func() ) [cte]
%           gamma = kernel hyperparameter ( see kernel_func() ) [cte]
%   Output: 
%       PAR.
%           Cx = Attributes of output dictionary                [p x Nk]
%           Cy = Classes of  output dictionary                  [Nc x Nk]
%           Km = Kernel matrix of dictionary                    [Nk x Nk]
%           Kmc = Kernel Matrix for each class (cell)           [Nc x 1]
%           Kinv = Inverse Kernel matrix of dicitionary         [Nk x Nk]
%           Kinvc = Inverse Kernel Matrix for each class (cell) [Nc x 1]
%           score = used for prunning method                    [1 x Nk]
%           class_history = used for prunning method           	[1 x Nk]
%           times_selected = used for prunning method           [1 x Nk]

%% INITIALIZATIONS

% Get Hyperparameters

Dm = HP.Dm;         % Design Method
Us = HP.Us;         % Update Strategy
eta = HP.eta;       % Update rate

% Get Parameters

Dx = HP.Cx;         % Attributes of dictionary
Dy = HP.Cy;         % Classes of dictionary

% Get Data

xt = DATA.input;    % Attributes of sample
yt = DATA.output;   % Class of sample

%% ALGORITHM

if (Us ~= 0),
    
    % Get sequential class of sample
    [~,yt_seq] = max(yt);
    
    % Find nearest prototype input
    if (Dm == 1),
        win = prototypes_win(Dx,xt,HP);
    elseif (Dm == 2),
    	[~,Dy_seq] = max(Dy);
        Dx_c = Dx(:,Dy_seq == yt_seq);
        win_c = prototypes_win(Dx_c,xt,HP);
        win = prototypes_win(Dx,Dx_c(:,win_c),HP);
    end
    
    % Find nearest prototype output
    y_new = Dy(:,win);
    [~,y_new_seq] = max(y_new);
    
    % Update Closest prototype (new one)
    if (Us == 1),       % (WTA)
        x_new = Dx(:,win) + eta * (xt - Dx(:,win));
    elseif (Us == 2),	% (LVQ)
        if(yt_seq == y_new_seq),
            x_new = Dx(:,win) + eta * (xt - Dx(:,win));
        else
            x_new = Dx(:,win) - eta * (xt - Dx(:,win));
        end
    elseif (Us == 3),   % (Derivative of kernel cost funtion)
        x_new = Dx(:,win) + eta * kernel_diff(xt,Dx(:,win),HP);
    end
    
    % New data to be added
    DATAnew.input = x_new;
    DATAnew.output = y_new;
    
    % Remove prototype from dictionary
    HP = isknn_rem_sample(HP,win);
    
    % Add Updated prototype to dictionary
    HP = isknn_add_sample(DATAnew,HP);
end

%% FILL OUTPUT STRUCTURE

PAR = HP;

%% END