THREADS_PER_BLOCK = 1024;
TILE = THREADS_PER_BLOCK/8;

temp = elements(1:128, :);
n = temp';

for k=1:TILE
    temp = 0;
    for j=1:8
        for i=j:8
            idx = temp + i + 36*(k-1);
            if (n(i+8*(k-1)) >= n(j+8*(k-1)))
                iK(idx) = n(i+8*(k-1));
                jK(idx) = n(j+8*(k-1));
            else
                iK(idx) = n(j+8*(k-1));
                jK(idx) = n(i+8*(k-1));
            end
        end
        temp = temp + i-j;
    end
end
