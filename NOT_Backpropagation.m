%Training Set 
x = [0 1];
y = [1 0];

%Initial Weights
w12 = [0.1;0.2];

%Learning Rate 
alpha = .25;

%Feedforward Path
cost = zeros(20);
for k=1:20
  for i=1:2
    a1 = [1;x(i)];
    a2 = a1' * w12;
    del_we = (a2 - y(i)) .* a1;
    w12 = w12 - alpha .* del_we;
  endfor
  cost(k) = 0.5*((a2 - y(2))^2);
endfor
plot(cost);
disp(w12);