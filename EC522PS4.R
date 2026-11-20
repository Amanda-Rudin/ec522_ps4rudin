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

