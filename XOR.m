clear all;
%Ex-Or function Generaation
alpha = 0.1;
%Training Set
x = [0,0,1,1;
     0,1,0,1];
x = (x-0.5)./0.5;
%Original output
y = [0,1,1,0];

%Initial Weights
w12 = [0.1 0.2; 0.3 0.4; 0.5 0.6];
w23 = [0.7;0.8;0.9];

cost = zeros(4,80);

for i= 1:80
  for j= 1:4
    for k= 1:100
    %Feedforward Path
    a1 = [1;x(:,j)];    %Inserting bias
    z2 = a1' * w12;         %Layer 2
    a2 = 1 ./ (1+exp(-z2)); %Layer 2
    a2 = [1 a2]';       %Inserting bias
    z3 = a2' * w23;     %Layer 3
    a3 = 1 ./ (1+exp(-z3));
       
    %Backward Path
    del_w23 = ((a3 - y(j))* (a3^2) * exp(-z3)) * a2;
    del_w12 = [(((a3-y(j))*((a3*a2(2))^2)*w23(2))*exp(-(z3+z2(1)))*a1 ),(((a3-y(j))*((a3*a2(3))^2)*w23(3))*exp(-(z3+z2(2)))*a1 )];
    
    %Gradient Descent
    w12 = w12 - alpha .* del_w12;
    w23 = w23 - alpha .* del_w23;
  endfor
  cost(j,i) = 0.5*((a3 - y(j))^2);
  endfor
endfor
hold on;
plot(cost(1,:));
plot(cost(2,:));
plot(cost(3,:));
plot(cost(4,:));
plot(mean(cost),'*');
hold off;
     