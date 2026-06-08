
setwd("./analysis/note") # change the working directory so that pdf comes in right place.
getwd()


# Let's construct a normal sine function 
pdf("./plot/sine_fn.pdf")

N <- 500 # number of shample points, finite memory finite pixels
fs <- 100 # say 100Hz
t <- seq(0, N/fs, length.out = N)


f <- 5
a <- 1

w <- 2 * pi * f

x <- a * sin(w * t)

plot(t, x, type = "l", col = "steelblue", lwd = 2,
     main = "Sine Wave", xlab = "Time (s)", ylab = "Amplitude")

x <- fft(x)
print(x[1]) # it output's list of complex numbers, complex number itself is meaningless
magnitude <- Mod(x)  # sqrt(a^2 + b^2)
print(magnitude[1])

freq_axis <- (0:(N-1)) * (fs / N)
half <- 1:(N/2)  

magnitude_scaled <- (2 / N) * magnitude[half]

plot(freq_axis[half], magnitude_scaled,
     type = "l", col = "steelblue", lwd = 2,
     main = "Frequency Spectrum",
     xlab = "Frequency (Hz)",
     ylab = "Amplitude")
