library(tidyverse)
library(fpp3)
library(BVAR)
liquor_raw <- read_csv(
  "data/MRTSSM4453USN.csv",
  na = ".",
  col_types = cols(
    observation_date = col_date(format = "%Y-%m-%d"),
    MRTSSM4453USN = col_double()
  )
)

liquor <- liquor_raw |>
  transmute(Month = yearmonth(observation_date), sales = MRTSSM4453USN) |>
  as_tsibble(index = Month) |>
  filter(Month <= yearmonth("2019 Dec"))

has_gaps(liquor)
liquor |>
  autoplot(sales) +
  labs(
    title = "US Liquor Store Sales, 1992–2019",
    x = "Month",
    y = "Sales (millions of dollars)"
  )
liquor |>
  autoplot(log(sales)) +
  labs(
    title = "Log of US Liquor Store Sales, 1992–2019",
    x = "Month",
    y = "log(Sales)"
  )

##write-up - The seasonal trend is that sales peak in late dec, then starkly dip, the reach mid levels again for the majority of the year. The dip in sales is during Jan and Feb. The seasonal swings do grow with the level. This suggests that we should  use logs, as they will help make constant percent changes constant in size even as the level grows.  

liquor_fit <- liquor |>
  model(arima = ARIMA(log(sales)),
        ets   = ETS(log(sales)))

liquor_fit |> select(arima) |> report()
liquor_fit |> select(ets) |> report()

#The ARIMA model was chosen as an ARIMA(1,0,2)(2,1,1)[12] w/ drift on the log of sales. This means that there was one seasonal difference and no regular diference. 
#The ETS model is ETS(A,A,A). This means there are additive errors, additive trend, and additive seasonality. 

liquor_fit |> select(arima) |> gg_tsresiduals()

liquor_fit |>
  select(arima) |>
  augment() |>
  features(.innov, ljung_box, lag = 24, dof = 6)
##The residuals do not look like white noise. The p-value is essentially 0, meaning we reject the H0 that the residuals are white noise. This means there is autocorrelation left in the residuals. 
liquor_cv <- liquor |>
  stretch_tsibble(.init = 120, .step = 1) |>
  filter(.id < max(.id))

liquor_cv_fit <- liquor_cv |>
  model(snaive = SNAIVE(sales),
        ets    = ETS(log(sales)),
       arima  = ARIMA(log(sales)))

saveRDS(liquor_cv_fit, "results/liquor_cv_fit.rds")
#My sample is Jan 1992 to Dec 2019, which is 336 months. Since the first training sample is 120 months, this means there are 216 forecast origins.
liquor_cv_fc <- liquor_cv_fit |>
  forecast(h = 1)

liquor_cv_fc |>
  accuracy(liquor, measures = list(RMSE = RMSE, MAE = MAE))
#The first forecast is Jan 2002 (the training sample is 120 months, so Jan 1992-Dec 2001, meaning one month ahead is Jan 2002). The last forecast is Dec 2019, as that is the last data point we have, so the last training sample ends in Nov 2019.
#The ARIMA model is the most accurate, with the lowest RMSE and MAE. Botg metrics agree - ARIMA has the lowest of both, ETS the next smallest, SNAIVE the largest. 

liquor_errs <- liquor_cv_fc |>
  as_tibble() |>
  left_join(as_tibble(liquor) |> select(Month, actual = sales),
            by = "Month") |>
  mutate(e = actual - .mean) |>
  select(Month, .model, e) |>
  pivot_wider(names_from = .model, values_from = e)

forecast::dm.test(liquor_errs$arima, liquor_errs$ets, h = 1, power = 2)


forecast::dm.test(liquor_errs$arima, liquor_errs$snaive, h = 1, power = 2)

#First test (arima v ets) has a DM stat of -2.9062 and a p-value of 0.004042 The second test (arima vs snaive) has a DM stat of -10.755 and a p-value of essentially 0. From this, we can conclude that both tests reject equal accuracy. A DM H0 is that the models are equally accurate, and we reject H0 for both test. We can also conclude that arima is more accurate than ets and snaive, as both DM stats are negative. If I were a liquor store chain, I would use the arima model.   

liquor_combo_fc <- liquor_cv_fit |>
  mutate(combo = (arima + ets) / 2) |>
  forecast(h = 1)

liquor_combo_fc |>
  accuracy(liquor, measures = list(RMSE = RMSE, MAE = MAE))

#Combo has a lower RMSE and MAE that both ets and snaive, but a higher RMSE and MAE than arima.

cor(liquor_errs$arima, liquor_errs$ets)

#The combo did not beat the best individual model. The errors are highly correlated (0.88). This means that arima and ets make similar mistakes. From the DM test, we know that arima is more accurate than ets. This means that combining them will do little to improve model accuracy, as the mistakes are similar and arima is more accurate, and equal weighting gives ets (the less accurate model) equal effect. This weighs down the combination's accuracy.

qd_raw <- read_csv("data/fredqd_2026-09.csv", show_col_types = FALSE) |>
  slice(-(1:2))

growth <- function(x) 100 * difference(log(x))

qd <- qd_raw |>
  mutate(date = as.Date(sasdate, "%m/%d/%Y")) |>
  transmute(date,
            gdp    = growth(GDPC1),     # real GDP
            cons   = growth(PCECC96),   # real consumption
            inv    = growth(GPDIC1),    # real investment
            emp    = growth(PAYEMS),    # payroll employment
            infl   = growth(GDPCTPI),   # GDP price index
            unrate = difference(UNRATE),     # unemployment rate
            ffr    = difference(FEDFUNDS),   # federal funds rate
            gs10   = difference(GS10)) |>    # 10-year Treasury yield
  filter(date >= as.Date("1960-03-01"), date <= as.Date("2019-12-01"))

y <- qd |> select(-date) |> as.matrix()

nrow(y)
head(y)

N <- nrow(y)
origins <- 120:(N - 1)
actual <- y[121:N, ]

#Each variable will have 120 forecasts produced, as we have 240 rows of data and our training sample is 120 quarters. 
#The first forecast date is 1990 Q1, and the last is 2019 Q4.

var_ols_fc <- function(z, p) {
  e <- embed(z, p + 1)            # columns: y_t, then its lags 1 to p
  M <- ncol(z)                    # number of variables
  B <- coef(lm(e[, 1:M] ~ e[, -(1:M)]))
  x <- c(1, t(z[nrow(z):(nrow(z) - p + 1), ]))   # constant, latest p lags
  x %*% B
}

fc_var1 <- NULL
fc_var4 <- NULL
for (T in origins) {
  fc_var1 <- rbind(fc_var1, var_ols_fc(y[1:T, ], 1))
  fc_var4 <- rbind(fc_var4, var_ols_fc(y[1:T, ], 4))
}

rmse_by_var <- function(f) sqrt(colMeans((actual - f)^2))

rmse_by_var(fc_var1)
rmse_by_var(fc_var4)

#The VAR(1) is better (lower RMSE) for 7 of the 8 variables, with only VAR(4) having a lower RMSE for inflation. Even though VAR(4) contains VAR(1), the smaller model might forecast better because the VAR(4) has to estimate a lot more coefficients. This leaves room for the extra lags to fit random noise, causing overfitting, which can worsen out-of-sample forecasts. This is a bias-variance tradeoff, where the VAR(1) has more bias but much less variance.

bvar_cv <- function(lambda) {
  pr <- bv_priors(hyper = "lambda",
                  mn = bv_mn(lambda = bv_lambda(mode = lambda, sd = 0.4,
                                                min = 0.999 * lambda,
                                                max = 1.001 * lambda),
                             b = 0))
  fc <- NULL
  for (T in origins) {
    fit_T <- bvar(y[1:T, ], lags = 4, priors = pr,
                  n_draw = 2000, n_burn = 1000, verbose = FALSE)
    x <- c(1, t(y[T:(T - 3), ]))         # constant and 4 lags of each variable
    fc <- rbind(fc, x %*% coef(fit_T))   # forecast of y_{T+1}
  }
  fc
}

set.seed(422)
fc_bvar_03 <- bvar_cv(0.3)
saveRDS(fc_bvar_03, "results/fc_bvar_03.rds")

set.seed(422)
fc_bvar_005 <- bvar_cv(0.05)

var_results <- tibble(
  Variable = colnames(y),
  `VAR(1)` = rmse_by_var(fc_var1),
  `VAR(4)` = rmse_by_var(fc_var4),
  `BVAR(4), lambda = 0.3` = rmse_by_var(fc_bvar_03),
  `BVAR(4), lambda = 0.05` = rmse_by_var(fc_bvar_005)
)

saveRDS(var_results, "results/var_results.rds")

var_results |> knitr::kable(digits = 3)

#1 - The BVAR(4) with lambda = 0.3 has a lower RMSE than the VAR(4) for all variables. Despite having the same lags and variables, the BVAR might forecast better because the Minnesota prior shrinkage helps get rid of noise in the coefficient estimates.
#2 - The BVAR(4) with lambda = 0.3 has a lower RMSE than VAR(1) for inv, infl,  and gs10. For the other variables, BVAR with lambda = 0.3 has a higher RMSE than the VAR(1). These are mixed results, showing that the VAR(1) is hard to beat. 
#3 - When lambda=0.05, the BVAR(4) has a higher RMSE than BVAR(4) with lambda = 0.3 for 7 of the 8 variables. The only exception is ffr. This tells us that lambda = 0.05 is too much shrinkage, and that the coefficients are shrunk to 0 too much so that they end up ignoring the info in the data. Since having shrinkage did beat the plain OLS VAR(4), we know that no shrinkage is too little, and that the VAR(4) is overfitting. If I used this table to choose my forecasting model, RMSE would not be a fair estimate for forecast accuracy, as it is the RMSE over this particular psuedo-out-of-sample. This means that our winning model wins over the data we have, but might not win over future data, because our winner fits our sample's randomness and noise best. This means the real RMSE of our winning model is likely higher than what the table shows. 

e_var4 <- actual[, "gdp"] - fc_var4[, 1]
e_bvar <- actual[, "gdp"] - fc_bvar_03[, 1]

forecast::dm.test(e_var4, e_bvar, h = 1, power = 2)

e_var4 <- actual[, "cons"] - fc_var4[, 2]
e_bvar <- actual[, "cons"] - fc_bvar_03[, 2]

forecast::dm.test(e_var4, e_bvar, h = 1, power = 2)

#From our DM tests, we can conclude that VAR4 is less accurate for GDP than BVAR w lambda=0.3. The DM test stat is 2.1366, which is positive, meaning the errors from VAR4 are larger. This test is signif at the 95% level.
#From our second DM test, we can conclude that BVAR w lambda=0.3 is more accurate than VAR4 for consumption. The DM test stat is positive again (3.0264), meaning teh errors for VAR4 are larger. This test is signif at the 95% level.
#This does not change our results in 4.3b because the results are consistent with our findings above. This test does show that the BVAR's lower RMSEs for GDP and Consumption are stat signif and not just luck of fitting to the noise. However, we only tested two variables.

md_raw <- read_csv("data/fredmd_2026-09.csv", show_col_types = FALSE)
tcode <- unlist(md_raw[1, -1])
md <- md_raw[-1, ] |> mutate(date = as.Date(sasdate, "%m/%d/%Y"))

transform_series <- function(x, code) {
  d <- function(v) c(NA, diff(v))
  switch(as.character(code),
         "1" = x,                      # level
         "2" = d(x),                   # first difference
         "3" = d(d(x)),                # second difference
         "4" = log(x),                 # log
         "5" = 100 * d(log(x)),        # growth rate
         "6" = 100 * d(d(log(x))),     # change in the growth rate
         "7" = d(x / dplyr::lag(x) - 1))
}

fm <- map2_dfc(md[names(tcode)], tcode, transform_series) |>
  mutate(date = md$date, .before = 1) |>
  filter(date >= as.Date("1960-03-01"), date <= as.Date("2019-12-01"))

fm <- fm[, colSums(is.na(fm)) == 0]

dim(fm)

tcode[c("INDPRO", "PAYEMS", "CPIAUCSL", "FEDFUNDS")]

#INDPRO - The series is transformed as a growth rate, or 100 x change in log.
#PAYEMS - The series is transformed as a growth rate, or 100 x change in log.
#CPIAUCSL - The series is transformed as a change in the growth rate.
#FEDFUNDS - The series is transformed as a first difference, or the change from last month.
#INDPRO is a total output index, so it gets a growth rate because it grows over time, meaning a growth rate helps make it stationary and more compaable. FEDFUNDS is already a percentage, being the interest rate, and can be near 0 meaning growth rates are not a good measure. Instead, we see the change in perecntage points month to month.

Z <- fm |> select(-date) |> as.matrix()

pc <- prcomp(scale(Z))

summary(pc)$importance[, 1:5]

#The first three factors explain 14.98% (PC1), 7.4%(PC2), and 7.06%(PC3) of the variance.
#In total, this is 29.44% in total.

F1 <- pc$x[, 1] * sign(cor(pc$x[, 1], fm$INDPRO))

tibble(date = fm$date, F1 = F1) |>
  ggplot(aes(date, F1)) +
  geom_line() +
  labs(title = "First Factor from FRED-MD", x = NULL, y = NULL) +
  theme_minimal()

sort(abs(cor(pc$x[, 1], Z))[1, ], decreasing = TRUE)[1:8]
sort(abs(cor(pc$x[, 2], Z))[1, ], decreasing = TRUE)[1:8]

#Factor 1 - real economic activity. The correlated series are things that measure employment and production. The plot confirms this, with drops during recessions, like in 1980 and 2008. It also seems to display business cycles. 
#Factor 2 - prices/inflation. The correlated series  are all different forms of CPIs.


