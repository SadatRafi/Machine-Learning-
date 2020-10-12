%Training Set 
x = [0 1];
y = [1 0];

%Initial Weights
w12 = [0.1;0.2];

%Learning Rate 
alpha = 2.5;

%Feedforward Path
cost = zeros(40);
for k=1:40
  for i=1:2
    a1 = [1;x(i)];                     %Inserting Bias
    a2 = 1 ./ (1+ exp(-(a1' * w12)));  %Second Layer Activation
    del_we = ((a2 .^2)*(a2 - y(i))*exp(-(a1' * w12))) .* a1;
    w12 = w12 - alpha .* del_we;
  endfor
  cost(k) = 0.5*((a2 - y(2))^2);
endfor
plot(cost);
disp(w12);