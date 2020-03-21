dat=csvread('../Result/1.csv');

% L=400;
% xtmp=[1:39]';
% 
% A=zeros(L,1);
% for i=1:L
%      A(i)=inv(xtmp'*xtmp)*xtmp'*dat(i,:)';
% end
% 
% plot(dat(1:400,:)');
% for i=1:L
%     z=A(i)*xtmp;
%     hold on
%     plot(z,'-.');
% end

L=200;
for i=1:L
    Ltmp=size(dat,2)
    z=zeros(1,Ltmp);
    z(1)=dat(i,1);
    for j=2:Ltmp
        z(j)=dat(i,j)-dat(i,j-1);
    end
    plot(z)
    hold on
end
