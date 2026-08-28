function [V, ge, gi] = t4_simple_wrap(params,stim_type,val_in,width_in,dur_in,pos_vect_in,time,fr,stimIdx_in)
%{ 
T5 Simple Model

Output
    V:      t x 1 vector with a simulation of T4 response to stimulus for a time
            vector t

Input
    params:     a 1 x 29 vector of parameters in order [mu, sig, A, Tr, Td, me,
                be, bi, Ti]. Each parameter is a block fo values for [E, I, E2, I2].
                ie) [muE, muI, muE2, muI2, sigE, sigI, ..., Ti]
    stim_type:  a string defining the stimulus type
                Options: 'spfr', 'mm', 'mov', 'edge'
    val:        Luminance of stimulus. 0 for dark relative to baseline, 1 for
                bright. paired cell with all stim_inputs
    width:      bar width. Expecting an integer value, typically in
                illuminated pixels. paired cell with all stim_inputs
    dur:        duration each step, in ms. Usually 40 or 160. paired cell with all stim_inputs
    pos_vect:   a vector indicating the Receptive Field-centered positioins
                to stimulate, in integers, indexed from leading edge. All positions will be stimulated
                for equal amounts of time except in moving bars. If
                stimulus is a moving bar moving in the preferred direction
                from position -4 to 4, pos_vet = {[-4:4]}. If ND, [4:-1:4].
                If minimal motion stimulus with flashes at -4 and 4,
                {-4,4}. paired cell with all stim_inputs
    time:       tx1 vector with entry values of time course, in ms. If the
                to simulate 1000ms sampled with a framerate of 0.5 ms,
                [0:0.5:1000].
    fr:         framerate, in ms.
    stimIdx:    logical of size t x 1, indexing when the next flash should
                be. sum(stimIdx) == length(pos_vect). note that for Gruntman T5 data
                moving bars are windowed, so sum(stimIdx) = length(pos_vect) = width-1.
                paired cell with all stim_inputs
%}
stim_curr          = 0;                % initialize a counter to finish loops when all stims are done
stim_tot           = length(val_in);   % finish repeating when we've done all stims
ge_mat             = zeros(stim_tot,length(time));     % preallocate to store conductances to each stimulus  
gi_mat             = zeros(stim_tot,length(time));

params_to_keep = who;
params_to_keep = who;
while stim_curr < stim_tot
    clearvars('-except', params_to_keep{:}) 
stim_curr = stim_curr+1; %go through each stim cond

spfr_data.val      = val_in{stim_curr};
spfr_data.width    = width_in{stim_curr};
spfr_data.stimDur  = dur_in{stim_curr};
spfr_data.pos_vect = pos_vect_in{stim_curr};
spfr_data.time     = time;
spfr_data.fr       = fr;
spfr_data.stimIdx  = stimIdx_in{stim_curr};
    
switch stim_type
    case 'spfr'
        spfr_data.stim_type = 1;
    case 'mm'
        spfr_data.stim_type = 2;
    case 'mov'
        spfr_data.stim_type = 3;
    case 'edge'
        spfr_data.stim_type = 4;
end

if spfr_data.val == 1                   %all params stored in "rows" of 4, 
    mu_p  = params(1:4);
    sig_p = params(5:8);
    amp_p = params(9:12);
    tr_p  = params([13,14,13,16])*10;
    td_p  = params([17,18,17,20])*10;
    me_p  = params(23)*.01;
    mi_p  = params(24)*.01;
    be_p  = params(27);
    bi_p  = params(28);
    ti_p  = params(29)*10;
else
    mu_p  = params([3:4,1:2]); %if NC, flip order of all param sets (seconds always fed into delta)
    sig_p = params([7:8,5:6]);
    amp_p = params([11:12,9:10]);
    tr_p  = params([13,16,13,14])*10;
    td_p  = params([17,20,17,18])*10;
    me_p  = params(21)*.01;
    mi_p  = params(22)*.01;
    be_p  = params(25);
    bi_p  = params(26);
    ti_p  = params(29)*10;
end

p.Vr = 0;
p.Ve = 65;
p.Vi = -10;

%stim parameters
width = spfr_data.width-1;
p.width = spfr_data.width;
p.stimDur = spfr_data.stimDur;
pos_vect = spfr_data.pos_vect;
d = 30/spfr_data.fr; %set manual delay to replicate delay in transmission from photoreceptors to T4/5 response
tmp = find(spfr_data.stimIdx);
stimIdx = false(size(spfr_data.stimIdx));
stimIdx(tmp+d) = true;
stim_time = fix(spfr_data.time(stimIdx));

buffer = 5;
M = 1:(length(spfr_data.pos_vect)+2*buffer);
x = M+min(spfr_data.pos_vect)-buffer;

Ie = zeros(length(stim_time),length(M));
Ii = zeros(length(stim_time),length(M));

for i = 1:length(pos_vect)
    pos = pos_vect(i) - min(pos_vect)+buffer;

     Ie(i,pos-width:pos) = 1;
     Ii(i,pos-width:pos) = 1;

end

p.Ie = Ie;
p.Ii = Ii;

if spfr_data.stim_type == 4
    Ie = cumsum(Ie,1) > 0;
    Ii = cumsum(Ii,1) > 0;
end

p.Ie = Ie;
p.Ii = Ii;

%excitatory (classic)
p.Tre =  tr_p(1);
p.Tde =  td_p(1);
Ae    = amp_p(1);
mue   =  mu_p(1);
sige  = sig_p(1);

%inhibitory (classic)
p.Tri =  tr_p(2);
p.Tdi =  td_p(2);
Ai    = amp_p(2);
mui   =  mu_p(2);
sigi  = sig_p(2); 

%excitatory2 (delta, from PC)
p.Tre2 =  tr_p(3);
p.Tde2 =  td_p(3);
Ae2    = amp_p(3);
mue2   =  mu_p(3);
sige2  = sig_p(3);

%inhibitory2 (delta, from PC)
p.Tri2 =  tr_p(4);
p.Tdi2 =  td_p(4);
Ai2    = amp_p(4);
mui2   =  mu_p(4);
sigi2  = sig_p(4);

%time
effDur = spfr_data.stimDur;
if spfr_data.stim_type>2
    effDur = effDur*spfr_data.width; %if moving, then also account for width 
end

%both contrasts are linear
p.be2 = max(0,effDur*me_p + be_p);
p.bi2 = max(0,effDur*mi_p + bi_p);

p.Tid = ti_p;

% Model
p.ae  =  Ae.*exp(-(x-mue).^2  / (2*sige^2) );
p.ai  =  Ai.*exp(-(x-mui).^2  / (2*sigi^2) );
p.ae2 = Ae2.*exp(-(x-mue2).^2 / (2*sige2^2) );
p.ai2 = Ai2.*exp(-(x-mui2).^2 / (2*sigi2^2) );

p.stim_time = stim_time;
t_ind = find([stimIdx;1]);
[~,s_ind] = min(abs(spfr_data.time - (spfr_data.time(stimIdx) + effDur)'));
s_ind = s_ind';

p.stim_time = stim_time;

%populate a matrix with this dynamic for each stim
fe = zeros(size(p.Ie,2),size(spfr_data.time,1));
fi = zeros(size(p.Ii,2),size(spfr_data.time,1));
fe2 = zeros(size(p.Ii,2),size(spfr_data.time,1));
fi2 = zeros(size(p.Ii,2),size(spfr_data.time,1));

switch spfr_data.stim_type
    case 1 %spfr
        %subtract delay from ending index to not encroach on subsequent SPFR
        t_tot = spfr_data.time(1:t_ind(2)-1);
        t_on = t_ind(1);
        t_off= s_ind(1);

        fe_tmp = fc_sol(t_tot, t_on, t_off, p.Tre, p.Tde);
        fi_tmp = fc_sol(t_tot, t_on, t_off, p.Tri, p.Tdi);

        fe2_tmp = fd_sol(t_tot, t_on, t_off, p.Tid, p.Tre2, p.Tde2, p.be2);
        fi2_tmp = fd_sol(t_tot, t_on, t_off, p.Tid, p.Tri2, p.Tdi2, p.bi2);


        %after solving for conductances at each time, we can now shift back
        %to non-delayed frame to fill vector
        t_ind = t_ind-d;
        s_ind = s_ind-d;
        for i = 1:size(Ie,1)
            fe(logical(p.Ie(i,:)),t_ind(i):(t_ind(i)+length(fe_tmp)-1)) = repmat(fe_tmp,sum(p.Ie(i,:)),1);
            fi(logical(p.Ii(i,:)),t_ind(i):(t_ind(i)+length(fi_tmp)-1)) = repmat(fi_tmp,sum(p.Ii(i,:)),1);
            fe2(logical(p.Ii(i,:)),t_ind(i):(t_ind(i)+length(fe2_tmp)-1)) = repmat(fe2_tmp,sum(p.Ii(i,:)),1);
            fi2(logical(p.Ii(i,:)),t_ind(i):(t_ind(i)+length(fe2_tmp)-1)) = repmat(fi2_tmp,sum(p.Ii(i,:)),1);
        end
    case 2
        t_tot = spfr_data.time;
        t_on = t_ind(1);
        t_off= s_ind(1);

        fe_tmp = fc_sol(t_tot, t_on, t_off, p.Tre, p.Tde);
        fi_tmp = fc_sol(t_tot, t_on, t_off, p.Tri, p.Tdi);

        fe2_tmp = fd_sol(t_tot, t_on, t_off, p.Tid, p.Tre2, p.Tde2, p.be2);
        fi2_tmp = fd_sol(t_tot, t_on, t_off, p.Tid, p.Tri2, p.Tdi2, p.bi2);
        
        %after solving for conductances at each time, we can now shift back
        %to non-delayed frame to fill vector
        t_ind = t_ind-d;
        s_ind = s_ind-d;

        for i = 1:size(Ie,1)
             fe(logical(p.Ie(i,:)),:)  = repmat(fe_tmp,sum(p.Ie(i,:)),1);
             fi(logical(p.Ii(i,:)),:)  = repmat(fi_tmp,sum(p.Ii(i,:)),1);
            fe2(logical(p.Ii(i,:)),:)  = repmat(fe2_tmp,sum(p.Ii(i,:)),1);
            fi2(logical(p.Ii(i,:)),:)  = repmat(fi2_tmp,sum(p.Ii(i,:)),1);
        end
           
    case 3 %moving bar
        %subtract delay from ending index to not encroach on subsequent SPFR
        %spfr_data.time = spfr_data.time - spfr_data.time(t_ind(1)); %because our analytical solutions assume t0 = 0
        t_tot = spfr_data.time;

        t_ind = [t_ind(1)*ones(width,1);t_ind];
        effDur = spfr_data.stimDur;
        effDur = effDur*sum(p.Ie,1); %if moving, then also account for width 
        fe_tmp = zeros(length(unique(effDur)),length(t_tot));
        fi_tmp = zeros(length(unique(effDur)),length(t_tot));
        fe2_tmp = zeros(length(unique(effDur)),length(t_tot));
        fi2_tmp = zeros(length(unique(effDur)),length(t_tot));

        %after solving for conductances at each time, we can now shift back
        %to non-delayed frame to fill vector
        t_ind = t_ind-d;
        s_ind = s_ind-d;

        counter2 = 0;
        for effDuri = unique(effDur)
            counter2 = counter2+1;

            be2 = max(0,effDuri*me_p + be_p); %if PC, deltas are constants.
            bi2 = max(0,effDuri*mi_p + bi_p); 

            [~,s] = min(abs(spfr_data.time - spfr_data.time(t_ind(1)) - effDuri));
            t_on = t_ind(1);
            t_off = s;
            fe_tmp(1 + effDuri/p.stimDur,:)  = fc_sol(t_tot, t_on, t_off, p.Tre, p.Tde);
            fi_tmp(1 + effDuri/p.stimDur,:)  = fc_sol(t_tot, t_on, t_off, p.Tri, p.Tdi);
            fe2_tmp(1 + effDuri/p.stimDur,:) = fd_sol(t_tot, t_on, t_off, p.Tid, p.Tre2, p.Tde2, be2);
            fi2_tmp(1 + effDuri/p.stimDur,:) = fd_sol(t_tot, t_on, t_off, p.Tid, p.Tri2, p.Tdi2, bi2);        
        end

        t_ind = t_ind - t_ind(1)+1;
        ii = 0;
        for i = 1:size(p.Ie,2)
            if ~any(p.Ie(:,i),1)
                continue
            end
            ii = ii+1;  
            fe(i,t_ind(ii):end)  =  fe_tmp(1+effDur(i)/p.stimDur, 1:end-t_ind(ii)+1);
            fi(i,t_ind(ii):end)  =  fi_tmp(1+effDur(i)/p.stimDur, 1:end-t_ind(ii)+1);
            fe2(i,t_ind(ii):end) = fe2_tmp(1+effDur(i)/p.stimDur, 1:end-t_ind(ii)+1);
            fi2(i,t_ind(ii):end) = fi2_tmp(1+effDur(i)/p.stimDur, 1:end-t_ind(ii)+1);
        end

        %after solving for conductances at each time, we can now shift back
        %to non-delayed frame to fill vector
        t_ind = t_ind-d;
        s_ind = s_ind-d;

        dd = mean(diff(pos_vect)) > 0;
        if ~dd %if ND, we need to reverse order
            fe(any(fe,2),:) = flipud(fe(any(fe,2),:));
            fi(any(fi,2),:) = flipud(fi(any(fi,2),:));
            fe2(any(fe2,2),:) = flipud(fe2(any(fe2,2),:));
            fi2(any(fi2,2),:) = flipud(fi2(any(fi2,2),:));
        end
    
    case 4 %edge
         %subtract delay from ending index to not encroach on subsequent SPFR
        %spfr_data.time = spfr_data.time - spfr_data.time(t_ind(1)); %because our analytical solutions assume t0 = 0
        t_tot = spfr_data.time;
        
        for i = 1:length(t_ind)-1
            effDur = spfr_data.stimDur*sum(p.Ie(:,i));
            t_on = t_ind(i)-1;
            t_off= s_ind(end);
            p.be2 = max(0,effDur*me_p + be_p);
            p.bi2 = max(0,effDur*mi_p + bi_p);
            fe(i+buffer,:)  = fc_sol(t_tot, t_on, t_off, p.Tre, p.Tde);
            fi(i+buffer,:)  = fc_sol(t_tot, t_on, t_off, p.Tri, p.Tdi);
            fe2(i+buffer,:) = fd_sol(t_tot, t_on, t_off, p.Tid, p.Tre2, p.Tde2, p.be2);
            fi2(i+buffer,:) = fd_sol(t_tot, t_on, t_off, p.Tid, p.Tri2, p.Tdi2, p.bi2);
        end

        dd = mean(diff(pos_vect)) > 0;
        if ~dd %if ND, we need to reverse order
            fe  = flipud(fe);
            fi  = flipud(fi);
            fe2 = flipud(fe2);
            fi2 = flipud(fi2);
        end
        
end

    fe = fe(:,1:size(spfr_data.time,1));
    fi = fi(:,1:size(spfr_data.time,1));
    fe2 = fe2(:,1:size(spfr_data.time,1));
    fi2 = fi2(:,1:size(spfr_data.time,1));
    
     %caluclate conductances and steady state voltage
    ge_mat(stim_curr,:)  = p.ae*fe + p.ae2*fe2;
    gi_mat(stim_curr,:)  = p.ai*fi + p.ai2*fi2;
end
    
   ge = sum(ge_mat,1);   %sum together all traces over the time course
   gi = sum(gi_mat,1);
    V = ((p.Vr + ge*p.Ve + gi*p.Vi)./(1+ge+gi))';
    V = V(1:size(spfr_data.time,1));
    ge = ge(1:size(spfr_data.time,1));
    gi = gi(1:size(spfr_data.time,1));


    function f = fc_sol(t_tot, t_on, t_off, Tr, Td)

        t0 = t_tot(t_on);
        t_tot = t_tot - t0; %shift everything so t0 is 0
        t1 = t_tot(t_off);  %so now t1 is effDur

        %solve where t0 = 0, so shift everything over
        F1 = @(t) (Td - Tr + Tr.*exp(-t./Tr))./(Td - Tr) - (Td.*exp(-t./Td))./(Td - Tr);
        F2 = @(t) exp(t1./Td).*exp(-t./Td).*((Td - Tr + Tr.*exp(-t1./Tr))./(Td - Tr) - (Td.*exp(-t1./Td))./(Td - Tr) + (Tr.*exp(t1./Td - t1./Tr).*exp(-t1./Td).*(exp(t1./Tr) - 1))./(Td - Tr)) - (Tr.*exp(t./Td - t./Tr).*exp(-t./Td).*(exp(t1./Tr) - 1))./(Td - Tr);
        f = [zeros(size(t_tot(1:t_on-1)));F1(t_tot(t_on:t_off-1));F2(t_tot(t_off:end))]';

        if any(isnan(f) | isinf(f))
            %if nans, solve the long way. t0 is 0, and t1 is effDur
                H1 = @(t) 1 - exp(-t./Tr);
                %F1 defined above
                
                B = H1(t1);
                C = F1(t1);

                t_tot = t_tot-t1; %shift frame where t1 is 0
                t1 = 0;
                H2 = @(t) B.*exp(-t./Tr);
                F2 = @(t) exp(-t./Td).*(C + (B.*Tr)./(Td - Tr)) - (B.*Tr.*exp(-t./Tr))./(Td - Tr);

            counter = 0;
            t_true = 0;
            while any(isnan(f) | isinf(f)) %if still nans, evaluate each ODE
                counter = counter+1;
                if counter> 100 %no unchecked while loops in cluster
                    break
                end            
                tmp_idx = find(isnan(f) | isinf(f),1) -1;  %find where function NaNs

                t_true = t_true + t1; %make t1 now relative to the old t1, so it is the time of integration until NaN
                t1 = t_tot(tmp_idx) - t_true; %t1 is integration time
                B = H2(t1);
                C = F2(t1);

                H2 = @(t) B.*exp(-t./Tr);
                F2 = @(t) exp(-t./Td).*(C + (B.*Tr)./(Td - Tr)) - (B.*Tr.*exp(-t./Tr))./(Td - Tr);
                f(tmp_idx+1:end) = F2(t_tot(tmp_idx+1:end) - t_tot(tmp_idx)); %add to output vector

            end %repeat while we have nans

        end
    end

    function f = fd_sol(t_tot, t_on, t_off, Ti, Tr, Td, b)

        t0 = t_tot(t_on);
        t_tot = t_tot - t0;
        t1 = t_tot(t_off);

        A = b;

        t_tot = t_tot - t1; %shift frame again so that t1 is 0
        t1 = 0;

        F2 = @(t) - (exp(-t./Td).*((A.*Td.*Ti^2.*exp(t./Td - t./Ti))./(Td - Ti) - (A.*Td.*Ti.*Tr.*exp(t./Td - t./Tr))./(Td - Tr)))./(Td.*Ti - Td.*Tr) - (A.*Td.*Ti.*exp(-t./Td))./(Td.*Ti + Td.*Tr - Ti.*Tr - Td^2);
        f = [zeros(size(t_tot(1:t_off-1)));F2(t_tot(t_off:end))]';

        if any(isnan(f) | isinf(f))
            %if nans define all functions explicitly and solve piecewise
            A = b;
            B = 0;
            C = 0;
            I2 = @(t) A.*exp(-t./Ti);
            H2 = @(t) exp(-t./Tr).*(B - (A.*Ti)./(Ti - Tr)) + (A.*Ti.*exp(-t./Ti))./(Ti - Tr);
            F2 = @(t) exp(-t./Td).*(C - ((Td.*Tr.*(A.*Ti - B.*Ti + B.*Tr))./(Td - Tr) - (A.*Td.*Ti^2)./(Td - Ti))./(Td.*Ti - Td.*Tr)) - (exp(-t./Td).*((A.*Td.*Ti^2.*exp(t./Td - t./Ti))./(Td - Ti) - (Td.*Tr.*exp(t./Td - t./Tr).*(A.*Ti - B.*Ti + B.*Tr))./(Td - Tr)))./(Td.*Ti - Td.*Tr);
            f = [zeros(size(t_tot(1:t_off-1)));F2(t_tot(t_off:end))]';
            
            counter = 0;
            t_true = t1;
            while any(isnan(f) | isinf(f))
                counter = counter+1;
                if counter> 100 %no unchecked while loops in cluster
                    break
                end            
                tmp_idx = find(isnan(f) | isinf(f),1) -1;  %find where function NaNs
                t_true = t_true+t1; %keep running index of last stopping point
                t1 = t_tot(tmp_idx) - t_true; %t1 is integration time, so difference from current stopping point and last stopping point

                A = I2(t1); %evaluate funciton just before nans. these are new init conds to decay
                B = H2(t1);
                C = F2(t1);

                I2 = @(t) A.*exp(-t./Ti); %these functions evaluate from 0
                H2 = @(t) exp(-t./Tr).*(B - (A.*Ti)./(Ti - Tr)) + (A.*Ti.*exp(-t./Ti))./(Ti - Tr);
                F2 = @(t) exp(-t./Td).*(C - ((Td.*Tr.*(A.*Ti - B.*Ti + B.*Tr))./(Td - Tr) - (A.*Td.*Ti^2)./(Td - Ti))./(Td.*Ti - Td.*Tr)) - (exp(-t./Td).*((A.*Td.*Ti^2.*exp(t./Td - t./Ti))./(Td - Ti) - (Td.*Tr.*exp(t./Td - t./Tr).*(A.*Ti - B.*Ti + B.*Tr))./(Td - Tr)))./(Td.*Ti - Td.*Tr);

                f(tmp_idx+1:end) = F2(t_tot(tmp_idx+1:end) - t_tot(tmp_idx)); %add to output vector
            end %repeat while we have nans

        end
    end

end