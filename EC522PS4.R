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





