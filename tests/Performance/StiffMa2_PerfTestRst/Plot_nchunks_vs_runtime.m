% Plot

%% Data
[xs,ys] = nchunks_vs_runtime('StiffMa2_LNX_Sca_sf180_nel180.mat', 1, 3);
[xv,yv] = nchunks_vs_runtime('StiffMa2_LNX_Vec_sf180_nel90.mat', 3, 3);

%% Curve fitting
n = 1; % 1.order polynomial(linear regression)
[ps, Ss] = polyfit(xs, ys, n);
[ys_fit,delta] = polyval(ps,xs,Ss);
% plot(xs,ys,'bo')
% hold on
% plot(xs,ys_fit,'r-')
% plot(xs,ys_fit+2*delta,'m--',xs,ys_fit-2*delta,'m--')
% title('Linear Fit of Data with 95% Prediction Interval')
% legend('Data','Linear Fit','95% Prediction Interval')

[pv, Sv] = polyfit(xv, yv, n);
[yv_fit,deltav] = polyval(pv,xv,Sv);
% plot(xv,yv,'bo')
% hold on
% plot(xv,yv_fit,'r-')
% plot(xv,yv_fit+2*deltav,'m--',xv,yv_fit-2*deltav,'m--')
% title('Linear Fit of Data with 95% Prediction Interval')
% legend('Data','Linear Fit','95% Prediction Interval')


%% Plot figure
figure1 = figure('color',[1,1,1]);
axes1 = axes('Parent',figure1);

pt = plot(xs,ys,'+k',xs,ys_fit,'-k',... % Scalar
    xv,yv,'*b',xv,yv_fit,':b');       % Vector

pt(1).LineWidth = 2;
pt(2).LineWidth = 2;
pt(3).LineWidth = 2;
pt(4).LineWidth = 2;

% Labels
xlabel('Number of chunks');
ylabel('Runtime (s)');

% Create legend
str1 = ['Scalar curve fitting: y = ',num2str(ps(1)),'x + ',num2str(ps(2))];
str2 = ['Vector curve fitting: y = ',num2str(pv(1)),'x + ',num2str(pv(2))];
legend('Scalar problem',str1,'Vector problem',str2);
legend1 = legend(axes1,'show');
set(legend1, 'NumColumns',1,'Location','northwest','Interpreter','latex');

% Set the remaining axes properties
box(axes1,'on');
% ylim(axes1,[0.001 100000]);
set(axes1,'XGrid','on','XMinorTick','on','XScale','linear',...
    'YGrid','on','YMinorTick','on','YScale','linear');


