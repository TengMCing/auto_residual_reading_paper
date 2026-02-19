## ----setup, include=FALSE-------------------
knitr::opts_chunk$set(
  message = FALSE, 
  warning = FALSE, 
  echo = FALSE,
  fig.width = 8,
  fig.height = 6,
  out.width = "100%", 
  fig.align = "center")


## -------------------------------------------
if (!requireNamespace("tidyverse", quietly = TRUE))
  install.packages("tidyverse")

if (!requireNamespace("glue", quietly = TRUE))
  install.packages("glue")

if (!requireNamespace("remotes", quietly = TRUE))
  install.packages("remotes")

# Visual inference models and p-value calculation
if (!requireNamespace("visage", quietly = TRUE)) 
  remotes::install_github("TengMCing/visage")

if (!requireNamespace("reticulate", quietly = TRUE))
  install.packages("reticulate")

if (!requireNamespace("ggmosaic", quietly = TRUE))
  install.packages("ggmosaic")

if (!requireNamespace("here", quietly = TRUE))
  install.packages("here")

if (!requireNamespace("knitr", quietly = TRUE))
  install.packages("knitr")

if (!requireNamespace("yardstick", quietly = TRUE))
  install.packages("yardstick")

if (!requireNamespace("kableExtra", quietly = TRUE))
  install.packages("kableExtra")

if (!requireNamespace("autovi", quietly = TRUE))
  install.packages("autovi")

if (!requireNamespace("patchwork", quietly = TRUE))
  install.packages("patchwork")

if (!requireNamespace("datasauRus", quietly = TRUE))
  install.packages("datasauRus")

# To install Python and the required libraries, you may use the following code chunk.
if (FALSE) {
  # Install `miniconda`
  # Skip if `conda` exists in the system
  if (is.null(reticulate:::find_conda()[[1]])) {
    reticulate::install_miniconda()
  }
  
  # You could use `options(reticulate.conda_binary = "/path/to/conda")` to
  # force `reticulate` to use a particular `conda` binary
  
  # Create an environment
  if (reticulate::condaenv_exists("auto_residual_reading_paper")) {
    reticulate::conda_remove("auto_residual_reading_paper")
  }
  
  reticulate::conda_create("auto_residual_reading_paper",
                           python_version = "3.11.9")
  
  reticulate::conda_install("auto_residual_reading_paper",
                            pip = TRUE,
                            packages = c("tensorflow"))
}

# Please use `reticulate::use_python()`, `reticulate::use_virtualenv()` or `reticulate::use_conda()` to specify your Python environment.
if (FALSE) {
  reticulate::use_condaenv("auto_residual_reading_paper")
}

if (!reticulate::py_module_available("tensorflow"))
  stop("Please specify your Python environment and ensure `tensorflow` is installed!")

if (!reticulate::py_module_available("PIL"))
  stop("Please specify your Python environment and ensure `PIL` is installed!")

options(tinytex.clean = FALSE)
library(tidyverse)
library(visage)
library(glue)
library(here)

# To control the simulation in this file
set.seed(10086)


## ----false-finding, fig.pos = "!h", fig.width = 4, fig.height = 4*4/5, fig.cap="An example residual vs fitted values plot (red line indicates 0 corresponds to the x-intercept, i.e. $y=0$). The vertical spread of the data points varies with the fitted values. This often indicates the existence of heteroskedasticity, however, here the result is due to skewed distribution of the predictors rather than heteroskedasticity. The Breusch-Pagan test rejects this residual plot at 95\\% significance level ($p\\text{-value} = 0.046$).", out.width = "50%"----
set.seed(452)
ori_x <- rand_lognormal()
mod <- heter_model(b = 0, x = closed_form(~-ori_x))
ori_dat <- mod$gen(300)

ori_dat %>%
  VI_MODEL$plot(theme = theme_light(), remove_grid_line = TRUE) +
  xlab("Fitted values") +
  ylab("Residuals")


## ----ex-lineup, fig.height = 6/4, fig.pos = "!h", fig.cap="An example lineup embedding the true residual plot among four null plots. In practice, lineups typically include 19 null plots, but a reduced set is shown here for presentation purposes. The null plots are generated via residual rotation to ensure consistency with $H_0$. Observers who have not previously seen the lineup are asked to identify the plot that appears most different. Under $H_0$ that the regression model is correctly specified, the true residual plot should be indistinguishable from the null plots, yielding a selection probability of 0.2. A small $p$-value arises when a substantial proportion of observers select the true residual plot (shown at position 2, exhibiting non-linearity)."----
set.seed(455)
mod <- poly_model()
ori_dat <- mod$gen_lineup(300, k = 5)

ori_dat %>%
  VI_MODEL$plot(theme = theme_light(), 
                remove_grid_line = TRUE,
                remove_axis = TRUE) +
  xlab("Fitted values") +
  ylab("Residuals") +
  facet_wrap(~k, ncol = 5)


## ----cnn-diag, fig.pos = "!h", fig.cap = "Diagram of the architecture of the optimized computer vision model. Numbers at the bottom of each box show the shape of the output of each layer. The band of each box drawn in a darker color indicates the use of the rectified linear unit activation function.  Yellow boxes are 2D convolutional layers, orange boxes are pooling layers, the grey box is the concatenation layer, and the purple boxes are dense layers.", out.width = "100%"----
knitr::include_graphics("figures/cnn.pdf")


## -------------------------------------------
test_pred <- read_csv(here("data/test_pred.csv"))
train_pred <- read_csv(here("data/train_pred.csv"))
meta <- read_csv(here("data/meta.csv"))

test_summary <- test_pred %>% 
  left_join(meta) %>%
  group_by(res) %>%
  summarise(RMSE = yardstick::rmse_vec(effect_size, vss),
            R2 = yardstick::rsq_vec(effect_size, vss),
            MAE = yardstick::mae_vec(effect_size, vss),
            HUBER = yardstick::huber_loss_vec(effect_size, vss)) %>%
  mutate(type = "test")

train_summary <- train_pred %>% 
  left_join(meta) %>%
  group_by(res) %>%
  summarise(RMSE = yardstick::rmse_vec(effect_size, vss),
            R2 = yardstick::rsq_vec(effect_size, vss),
            MAE = yardstick::mae_vec(effect_size, vss),
            HUBER = yardstick::huber_loss_vec(effect_size, vss)) %>%
  mutate(type = "train")

model_pred <- train_pred %>% 
  left_join(meta) %>%
  mutate(type = "train") %>%
  bind_rows(test_pred %>% 
              left_join(meta) %>%
              mutate(type = "test"))

data_overall <- model_pred %>%
  filter(res == 32L) %>%
  group_by(include_non_normal, include_heter, include_z) %>%
  summarise(train_n = sum(type == "train"),
            train_RMSE = yardstick::rmse_vec(effect_size[type == "train"], vss[type == "train"]),
            test_n = sum(type == "test"),
            test_RMSE = yardstick::rmse_vec(effect_size[type == "test"], vss[type == "test"])) %>%
  mutate(violations = ifelse(include_z, "non-linearity", "null")) %>%
  mutate(violations = ifelse(include_heter, glue("{violations} + heteroskedasticity"), violations)) %>%
  mutate(violations = ifelse(include_non_normal, glue("{violations} + non-normality"), violations)) %>%
  mutate(violations = gsub("null \\+ ", "", violations)) %>%
  ungroup() %>%
  select(violations, train_n, train_RMSE, test_n, test_RMSE) %>%
  mutate(train_n = scales::comma(train_n), 
         test_n = scales::comma(test_n)) %>%
  mutate(across(c(train_RMSE, test_RMSE), ~format(.x, digits = 3))) %>%
  mutate(violations = ifelse(violations == "null", "no violations", violations))


## -------------------------------------------
data_overall %>%
  kableExtra::kable(format = "latex",
                    booktabs = TRUE,
                    label = "data-overall",
                    caption = "Number of training and test samples for each model violation scenario, including cases with multiple simultaneous violations. Each sample consists of a residual plot as input and the corresponding distance $D$ as the target. Root mean square error (RMSE) values for the training and test sets are shown for each scenario. Details of the synthetic data generating process used to construct these samples are provided in Appendix A.",
                    escape = FALSE,
                    linesep = "", 
                    align = "lrr",
                    col.names = c("Violations", "\\#samples", "RMSE", "\\#samples", "RMSE")) %>%
  kableExtra::add_header_above(c(" ", "Train" = 2, "Test" = 2)) %>%
  kableExtra::kable_styling(latex_options = "scale_down")


## -------------------------------------------
test_summary %>%
  select(res, RMSE, R2, MAE, HUBER) %>%
  mutate(res = glue("${res} \\times {res}$")) %>%
  mutate(across(RMSE:MAE, ~format(.x, digits = 3))) %>%
  mutate(across(HUBER, ~format(.x, digits = 2))) %>%
  kableExtra::kable(format = "latex",
                    booktabs = TRUE, 
                    label = "performance",
                    caption = "The test performance of three optimized models with different input sizes. The metrics are computed by comparing the estimated distance $\\hat{D}$ (model output) and the target distance $D$ generated from the synthetic data model. RMSE represents the root mean squared error; $R^2$ is the squared correlation between the two quantities; MAE denotes the mean absolute error; and Huber loss refers to the average Huber loss computed over the test set.",
                    escape = FALSE,
                    align = "lrrrr",
                    col.names = c("", "RMSE", "$R^2$", "MAE", "Huber loss"))


## ----model-performance, fig.pos = "!h", fig.cap = "Hexagonal heatmap for difference in $D$ and $\\hat{D}$ vs $D$ on test data for three optimized models with different input sizes. The brown lines are smoothing curves produced by fitting generalized additive models. Over-prediction and under-prediction can be observed for small $D$ and large $D$ respectively.", dev = 'png', dpi = 300, fig.height = 3----
model_pred %>%
  filter(type == "test") %>%
  ggplot() +
  geom_hline(yintercept = 0, alpha = 0.5) +
  annotate("rect", 
           ymin = min(model_pred$effect_size - model_pred$vss),
           ymax = 0,
           xmin = min(model_pred$effect_size),
           xmax = max(model_pred$effect_size),
            fill = "#40B0A6",
            alpha = 0.3) +
  geom_point(data = NULL, aes(5, 0, col = "over-prediction"), shape = 15) +
  annotate("rect", 
           ymin = 0,
           ymax = max(model_pred$effect_size - model_pred$vss),
           xmin = min(model_pred$effect_size),
           xmax = max(model_pred$effect_size),
           fill = "#E1BE6A",
           alpha = 0.3) +
  geom_point(data = NULL, aes(5, 0, col = "under-prediction"), shape = 15) +
  geom_hex(aes(effect_size, effect_size - vss), bins = 20) +
  geom_smooth(aes(effect_size, effect_size - vss), se = FALSE, col = "#994F00") +
  facet_wrap(~ res) +
  coord_fixed() +
  ylab(expression(D - hat(D))) +
  xlab(expression(D)) +
  theme_bw() +
  scale_fill_continuous(limits = c(1, 6000), trans = "log10", low = "#56B1F7", high = "#132B43") +
  scale_color_manual(values = c("under-prediction" = scales::alpha("#E1BE6A", 0.3),
                                "over-prediction" = scales::alpha("#40B0A6", 0.3)),
                     breaks = c("under-prediction", "over-prediction")) +
  guides(col = guide_legend(title = NULL, override.aes = list(size = 8), order = 1))


## ----over-under, fig.pos = "!h", fig.cap = "Scatter plots for difference in $D$ and $\\hat{D}$ vs $\\sigma$ on test data for the $32 \\times 32$ optimized model. The data is grouped by whether the regression has only non-linearity violation, and whether it includes a second predictor in the regression formula. The brown lines are smoothing curves produced by fitting generalized additive models. Under-prediction mainly occurs when the data-generating process has small $\\sigma$, a second predictor, and only non-linearity as the model violation.", dev = 'png', dpi = 300, fig.height = 4----
model_pred %>%
  filter(res == 32L) %>%
  filter(type == "test") %>%
  mutate(only_z = include_z & !include_heter & !include_non_normal) %>%
  mutate(only_z = ifelse(only_z, "Non-linearity only", "Multiple violations")) %>%
  mutate(include_x2 = ifelse(include_x2, "With second predictor", "With single predictor")) %>%
  rename(`Only has non-linearity violation` = only_z, `Has second predictor` = include_x2) %>%
  ggplot() +
  geom_hline(yintercept = 0, alpha = 0.5) +
  annotate("rect", 
           ymin = min(model_pred$effect_size[model_pred$type == "test"] - model_pred$vss[model_pred$type == "test"]),
                ymax = 0,
                xmin = min(model_pred$e_sigma),
                xmax = max(model_pred$e_sigma),
            fill = "#40B0A6",
            alpha = 0.3) +
  annotate("rect", 
           ymin = 0,
                ymax = max(model_pred$effect_size[model_pred$type == "test"] - model_pred$vss[model_pred$type == "test"]),
                xmin = min(model_pred$e_sigma),
                xmax = max(model_pred$e_sigma),
            fill = "#E1BE6A",
            alpha = 0.3) +
  geom_point(aes(e_sigma, effect_size - vss), alpha = 0.2) +
  geom_smooth(aes(e_sigma, effect_size - vss), se = FALSE, col = "#994F00") +
  facet_grid(`Has second predictor` ~ `Only has non-linearity violation`) +
  theme_bw() +
  ylab(expression(D - hat(D))) +
  xlab(expression(sigma))


## ----rd-human-------------------------------
vss_32 <- readRDS(here("data/vss_32.rds"))

experiment <- vi_survey %>%
  group_by(unique_lineup_id) %>%
  summarise(across(everything(), first)) %>%
  select(unique_lineup_id, attention_check, null_lineup, prop_detect,
         answer, effect_size, conventional_p_value,
         p_value, type, shape, a, b, x_dist, e_dist,
         e_sigma, include_z, k, n)

experiment <- vss_32 %>%
  left_join(experiment) %>%
  mutate(conventional_reject = conventional_p_value <= 0.05) %>%
  mutate(reject = p_value <= 0.05) %>%
  mutate(model_reject = vss_p_value <= 0.05) %>%
  mutate(model_boot_reject = vss_boot_p_value <= 0.05)

experiment <- experiment %>%
  left_join(readRDS(here("data/actual_ss.rds")))


## -------------------------------------------
experiment %>%
  filter(!attention_check) %>%
  filter(!null_lineup) %>%
  mutate(type = ifelse(type == "polynomial", "non-linearity", type)) %>%
  group_by(type) %>%
  summarise(RMSE = yardstick::rmse_vec(actual_ss, vss),
            R2 = yardstick::rsq_vec(actual_ss, vss),
            MAE = yardstick::mae_vec(actual_ss, vss),
            HUBER = yardstick::huber_loss_vec(actual_ss, vss)) %>%
  mutate(across(RMSE:HUBER, ~format(.x, digits = 3))) %>%
  kableExtra::kable(format = "latex",
                    booktabs = TRUE, 
                    label = "experiment-performance",
                    caption = "The performance of the $32 \\times 32$ model on the data used in the human subject experiment.",
                    escape = FALSE,
                    align = "lrrrr",
                    col.names = c("Violation", "RMSE", "$R^2$", "MAE", "Huber loss"))


## ----hist-null-human------------------------
lineup_vss <- readRDS(here("data/lineup_vss.rds"))
lineup_vss <- lineup_vss %>%
  group_by(unique_lineup_id) %>%
  mutate(delta_diff = vss[!null] - max(vss[null])) %>%
  mutate(gamma_diff = sum(vss[null] > vss[!null]))


experiment <- experiment %>%
  left_join(lineup_vss %>%
  filter(null == FALSE) %>%
  mutate(model_reject_20 = rank == 1) %>%
  select(unique_lineup_id, rank, model_reject_20))


## ----cache = TRUE---------------------------
vi_lineup <- get_vi_lineup()


## -------------------------------------------
bind_rows(
  experiment %>%
    filter(!attention_check) %>%
    filter(!null_lineup) %>%
    mutate(type = ifelse(type == "polynomial", "non-linearity", type)) %>%
    group_by(type) %>%
    summarise(n = format(n()), agree = format(sum(model_reject == conventional_reject)), rate = format(mean(conventional_reject == model_reject), digits = 4)),
  experiment %>%
    filter(!attention_check) %>%
    filter(!null_lineup) %>%
    mutate(type = ifelse(type == "polynomial", "non-linearity", type)) %>%
    group_by(type) %>%
    summarise(n = format(n()), agree = format(sum(model_reject == reject)), rate = format(mean(model_reject == reject), digits = 4))
  ) %>%
  kableExtra::kable(format = "latex",
                    booktabs = TRUE,
                    label = "human-conv-table",
                    caption = "Summary of the comparison of decisions made by computer vision model with decisions made by conventional tests and visual tests conducted by human.",
                    escape = FALSE,
                    align = "lrrr",
                    col.names = c("Violations", "\\#Samples", "\\#Agreements", "Agreement rate")) %>%
  kableExtra::pack_rows("Compared with conventional tests", 1, 2) %>%
  kableExtra::pack_rows("Compared with visual tests conducted by human", 3, 4)


## ----conv-mosaic, fig.pos = "!h", fig.cap = "Rejection rate ($p$-value $\\leq0.05$) of computer vision models conditional on conventional tests on non-linearity (left) and heteroskedasticity (right) lineups displayed using a mosaic plot. When the conventional test fails to reject, the computer vision mostly fails to reject the same plot as well as indicated by the height of the top right yellow rectangle, but there are non negliable amount of plots where the conventional test rejects but the computer vision model fails to reject as indicated by the width of the top left yellow rectangle.", fig.height = 3----
library(ggmosaic)

experiment %>%
  filter(!attention_check) %>%
  filter(!null_lineup) %>%
  mutate(type = ifelse(type == "polynomial", "non-linearity", type)) %>%
  mutate(type = ifelse(type == "non-linearity", "Non-linearity", "Heteroskedasticity")) %>%
  mutate(type = factor(type, levels = c("Non-linearity", "Heteroskedasticity"))) %>%
  mutate(model_reject_20 = ifelse(model_reject_20, "Reject", "No")) %>%
  mutate(conventional_reject = ifelse(conventional_reject, "Reject", "No")) %>%
  mutate(across(c(model_reject_20, conventional_reject), ~factor(.x, levels = c("Reject", "No")))) %>%
  ggplot() +
  geom_mosaic(aes(x = product(model_reject_20, conventional_reject), 
                  fill = model_reject_20)) +
  facet_grid(~type) +
  ylab("Computer vision model rejects") +
  xlab("Conventional tests reject ") +
  scale_fill_manual(values = c("#40B0A6", "#E1BE6A")) + 
  theme_bw() +
  theme(legend.position = "none") +
  coord_fixed()


## ----pcp, fig.pos = "!h", fig.cap = "Parallel coordinate plots of decisions made by computer vision model, conventional tests and visual tests made by human. All three agree in around 50% of cases. The model rejects less often than conventional tests when humans do not, and rarely rejects when both others do not, indicating closer alignment with human judgment.", eval = TRUE, fig.height = 3.5----
library(ggpcp)
mutate(experiment, type = ifelse(type == "polynomial", "Non-linearity", "Heteroskedasticity")) %>%
  mutate(type = factor(type, levels = c("Non-linearity", "Heteroskedasticity"))) %>%
  filter(!null_lineup, !attention_check) %>%
  select(type, actual_ss, model_reject_20, reject, conventional_reject) %>%
  mutate(test = rnorm(n())) %>%
  mutate(across(c(model_reject_20, conventional_reject, reject), ~ifelse(.x, "Reject", "Not reject"))) %>%
  mutate(agree = model_reject_20 == reject) %>%
  pcp_select(c(reject, model_reject_20, conventional_reject)) %>%
  group_by(type) %>%
  pcp_scale() %>%
  ungroup() %>%
  pcp_arrange(method = "from-right") %>%
  ggplot() +
  geom_pcp(aes_pcp(), linewidth = 0.1) +
  geom_pcp_labels(aes_pcp()) +
  geom_pcp_boxes(aes_pcp(), boxwidth = 0.1) +
  facet_wrap(~type, scales = "free_y") +
  theme_bw() +
  scale_x_discrete(labels = c("Visual\ntest", "Computer\nvision\nmodel", "Conventional\ntest")) +
  labs(col = "D") +
  ylab("") +
  xlab("") +
  theme(axis.text.y = element_blank(), 
        axis.ticks.y = element_blank(), 
        axis.title = element_blank())


## ----power, fig.pos = "!h", fig.cap = "Comparison of power of visual tests, conventional tests and the computer vision model. Marks along the x-axis at the bottom of the plot represent rejections made by each type of test. Marks at the top of the plot represent acceptances. Power curves are fitted by logistic regression models with no intercept but an offset equals to $\\text{log}(0.05/0.95)$. The model is less sensitive than conventional tests for small $D$ but similarly sensitive for large $D$. Visual tests are least sensitive overall. The model’s curve lies between those of conventional and visual tests.", fig.width = 7, fig.height = 3.5----

max_actual_ss <- max(filter(experiment, !attention_check, !null_lineup)$actual_ss)
max_poly_ss <- max(filter(experiment, !attention_check, !null_lineup, type == "polynomial")$actual_ss)
max_heter_ss <- max(filter(experiment, !attention_check, !null_lineup, type != "polynomial")$actual_ss)

model_glm_pred_poly <- glm(model_reject_20 ~ actual_ss - 1, 
    data = filter(experiment, !attention_check, !null_lineup, type == "polynomial") %>% mutate(offset0 = log(0.05/0.95)), 
    offset = offset0,
    family = binomial()) %>%
  predict(data.frame(actual_ss = seq(0, max_poly_ss, 0.01), offset0 = log(0.05/0.95)),
          type = "response") %>%
  data.frame(d = seq(0, max_poly_ss, 0.01), type = "Non-linearity", name = "Computer vision model", value = .) %>%
  mutate(type = factor(type, levels = c("Non-linearity", "Heteroskedasticity")))

visual_glm_pred_poly <- glm(reject ~ actual_ss - 1, 
    data = filter(experiment, !attention_check, !null_lineup, type == "polynomial") %>% mutate(offset0 = log(0.05/0.95)), 
    offset = offset0,
    family = binomial()) %>%
  predict(data.frame(actual_ss = seq(0, max_poly_ss, 0.01), offset0 = log(0.05/0.95)),
          type = "response") %>%
  data.frame(d = seq(0, max_poly_ss, 0.01), type = "Non-linearity", name = "Visual test", value = .) %>%
  mutate(type = factor(type, levels = c("Non-linearity", "Heteroskedasticity")))

conv_glm_pred_poly <- glm(conventional_reject ~ actual_ss - 1, 
    data = filter(experiment, !attention_check, !null_lineup, type == "polynomial") %>% mutate(offset0 = log(0.05/0.95)), 
    offset = offset0,
    family = binomial()) %>%
  predict(data.frame(actual_ss = seq(0, max_poly_ss, 0.01), offset0 = log(0.05/0.95)),
          type = "response") %>%
  data.frame(d = seq(0, max_poly_ss, 0.01), type = "Non-linearity", name = "Conventional test", value = .) %>%
  mutate(type = factor(type, levels = c("Non-linearity", "Heteroskedasticity")))

model_glm_pred_heter <- glm(model_reject_20 ~ actual_ss - 1, 
    data = filter(experiment, !attention_check, !null_lineup, type != "polynomial") %>% mutate(offset0 = log(0.05/0.95)), 
    offset = offset0,
    family = binomial()) %>%
  predict(data.frame(actual_ss = seq(0, max_heter_ss, 0.01), offset0 = log(0.05/0.95)),
          type = "response") %>%
  data.frame(d = seq(0, max_heter_ss, 0.01), type = "Heteroskedasticity", name = "Computer vision model", value = .) %>%
  mutate(type = factor(type, levels = c("Non-linearity", "Heteroskedasticity")))

visual_glm_pred_heter <- glm(reject ~ actual_ss - 1, 
    data = filter(experiment, !attention_check, !null_lineup, type != "polynomial") %>% mutate(offset0 = log(0.05/0.95)), 
    offset = offset0,
    family = binomial()) %>%
  predict(data.frame(actual_ss = seq(0, max_heter_ss, 0.01), offset0 = log(0.05/0.95)),
          type = "response") %>%
  data.frame(d = seq(0, max_heter_ss, 0.01), type = "Heteroskedasticity", name = "Visual test", value = .) %>%
  mutate(type = factor(type, levels = c("Non-linearity", "Heteroskedasticity")))

conv_glm_pred_heter <- glm(conventional_reject ~ actual_ss - 1, 
    data = filter(experiment, !attention_check, !null_lineup, type != "polynomial") %>% mutate(offset0 = log(0.05/0.95)), 
    offset = offset0,
    family = binomial()) %>%
  predict(data.frame(actual_ss = seq(0, max_heter_ss, 0.01), offset0 = log(0.05/0.95)),
          type = "response") %>%
  data.frame(d = seq(0, max_heter_ss, 0.01), type = "Heteroskedasticity", name = "Conventional test", value = .) %>%
  mutate(type = factor(type, levels = c("Non-linearity", "Heteroskedasticity")))

mutate(experiment, type = ifelse(type == "polynomial", "Non-linearity", "Heteroskedasticity")) %>%
  mutate(type = factor(type, levels = c("Non-linearity", "Heteroskedasticity"))) %>%
  filter(!null_lineup, !attention_check) %>%
ggplot() +
  geom_line(data = model_glm_pred_poly, aes(d, value, col = "Computer vision model"), linewidth = 0.8) +
  geom_line(data = model_glm_pred_heter, aes(d, value, col = "Computer vision model"), linewidth = 0.8) +
  geom_line(data = visual_glm_pred_poly, aes(d, value, col = "Visual test"), linewidth = 0.8) +
  geom_line(data = visual_glm_pred_heter, aes(d, value, col = "Visual test"), linewidth = 0.8) +
  geom_line(data = conv_glm_pred_poly, aes(d, value, col = "Conventional test"), linewidth = 0.8) +
  geom_line(data = conv_glm_pred_heter, aes(d, value, col = "Conventional test"), linewidth = 0.8) +
  geom_segment(data = ~filter(.x, !reject), aes(x = actual_ss, xend = actual_ss, y = -0.1, yend = -0.1 + 0.03, col = "Visual test"), alpha = 0.4) +
  geom_segment(data = ~filter(.x, reject), aes(x = actual_ss, xend = actual_ss, y = 1.1 - 0.03 * 2, yend = 1.1 - 0.03 * 3, col = "Visual test"), alpha = 0.4) +
  geom_segment(data = ~filter(.x, !model_reject_20), aes(x = actual_ss, xend = actual_ss, y = -0.1 + 0.03, yend = -0.1 + 0.03 * 2, col = "Computer vision model"), alpha = 0.4) +
  geom_segment(data = ~filter(.x, model_reject_20), aes(x = actual_ss, xend = actual_ss, y = 1.1 - 0.03, yend = 1.1 - 0.03 * 2, col = "Computer vision model"), alpha = 0.4) +
  geom_segment(data = ~filter(.x, !conventional_reject), aes(x = actual_ss, xend = actual_ss, y = -0.1 + 0.03 * 2, yend = -0.1 + 0.03 * 3, col = "Conventional test"), alpha = 0.4) +
  geom_segment(data = ~filter(.x, conventional_reject), aes(x = actual_ss, xend = actual_ss, y = 1.1, yend = 1.1 - 0.03, col = "Conventional test"), alpha = 0.4) +
  facet_grid(~type, scales = "free_x") +
  theme_bw() +
  scale_color_brewer("", palette = "Dark2") +
  theme(legend.position = "bottom") +
  xlab("D") +
  ylab("Reject")


## ----delta, fig.pos = "!h", fig.cap = "A weighted detection rate vs adjusted $\\delta$-difference plot. The brown line is smoothing curve produced by fitting generalized additive models. Detection generally increases with positive $\\delta_{\\text{adj}}$, but variability and exceptions highlight the distance measure's imperfect alignment with human perception.", fig.width = 6, fig.height = 3, out.width = "80%"----
experiment %>%
  left_join(lineup_vss %>%
              group_by(unique_lineup_id) %>%
              summarise(delta_diff = first(delta_diff), gamma_diff = first(gamma_diff))) %>%
  ggplot() +
  geom_vline(xintercept = 0, alpha = 0.4, linetype = 2) +
  geom_point(aes(delta_diff, prop_detect), alpha = 0.3) +
  geom_smooth(aes(delta_diff, prop_detect), se = FALSE, col = "#994F00") +
  ylab("Weighted detection rate") +
  xlab(expression(delta[adj])) +
  theme_light()


## -------------------------------------------
set.seed(452)
ori_x <- rand_lognormal()
dat <- heter_model(b = 0, x = closed_form(~-ori_x))$gen(300)

mod <- lm(y ~ x, data = dat)
my_vi <- autovi::auto_vi(fitted_model = mod)


## -------------------------------------------
if (!file.exists(here("cached_data/fig1_check.rds"))) {
  keras_model <- autovi::get_keras_model("vss_phn_32")
  my_vi$check(null_draws = 200L, boot_draws = 200L, keras_model = keras_model, extract_feature_from_layer = "global_max_pooling2d")
  saveRDS(my_vi$check_result, here("cached_data/fig1_check.rds"))
}



## -------------------------------------------
my_vi$check_result <- readRDS(here("cached_data/fig1_check.rds"))


## -------------------------------------------
define_gradient <- function() {
  reticulate::py_run_string(glue(
"
import tensorflow as tf
import PIL
import numpy as np

def get_smooth_gradient(keras_mod, plot_path, target_size, input_auxiliary = None, noise_level = 0.1, n = 50):
    im = PIL.Image.open(plot_path)
    im = im.resize(target_size)
    input_im = tf.keras.utils.img_to_array(im)
    input_im = np.reshape(input_im, tuple([1]) + tuple(target_size) + tuple([3]))
    if input_auxiliary is not None:
        input_auxiliary = np.reshape(np.array(input_auxiliary), (1, 5))
    
    if input_auxiliary is not None:
        input_auxiliary = tf.Variable(input_auxiliary)
    
    im_grad = None
    sigma = noise_level * (np.max(input_im) - np.min(input_im))
    
    for i in range(n):    
        new_input_im = input_im + np.random.normal(0, sigma, input_im.shape)
        new_input_im = tf.Variable(new_input_im)
        
        with tf.GradientTape() as tape:
            if input_auxiliary is not None:
                pred = keras_mod([new_input_im, input_auxiliary])
            else:
                pred = keras_mod(new_input_im)
        new_im_grad = tape.gradient(pred, new_input_im)
        new_im_grad = tf.image.rgb_to_grayscale(new_im_grad)[0].numpy()
        if im_grad is None:
            im_grad = new_im_grad
        else:
            im_grad = im_grad + new_im_grad
    
    im_grad = im_grad / n
    
    input_im = tf.Variable(input_im)
    if input_auxiliary is not None:
        with tf.GradientTape() as tape:
            pred = keras_mod([input_im, input_auxiliary])
        auxiliary_grad = tape.gradient(pred, input_auxiliary)
    if input_auxiliary is not None:
        return (im_grad, auxiliary_grad[0].numpy())
    else:
        return im_grad
"
))
}



## -------------------------------------------
if (!file.exists(here("cached_data/fig1_attention_map.rds"))) {
  reticulate::py_set_seed(10086, disable_hash_randomization = TRUE)
  define_gradient()
  keras_mod <- autovi::get_keras_model("vss_phn_32")
  plot_path <- my_vi$plot_resid() %>%
    autovi::save_plot()
  input_auxiliary <- my_vi$auxiliary()

  reticulate::py$get_smooth_gradient(keras_mod = keras_mod, 
                                     plot_path = plot_path, 
                                     target_size = c(32L, 32L), 
                                     input_auxiliary = unlist(input_auxiliary),
                                     noise_level = 0.5,
                                     n = 1000L) -> smooth_g
  
  x_range <- ggplot_build(my_vi$plot_resid())$layout$panel_params[[1]]$x.range
  y_range <- ggplot_build(my_vi$plot_resid())$layout$panel_params[[1]]$y.range
  
  smooth_att_map <- smooth_g[[1]] |> 
    as.data.frame() |>
    mutate(row = rev(1:32)) |>
    pivot_longer(V1:V32, names_to = "column", values_to = "gradient") |>
    mutate(column = as.integer(gsub("V", "", column))) |>
    mutate(column = x_range[1] + column * (x_range[2] - x_range[1])/(32 - 1)) |>
    mutate(row = y_range[1] + row * (y_range[2] - y_range[1])/(32 - 1))
  
  scale_zero_one <- function(x) (x - min(x))/(max(x) - min(x))
  
  p <- ggplot() +
    geom_raster(data = mutate(smooth_att_map, gradient = scale_zero_one(gradient)), aes(column, row, fill = gradient), alpha = 0.7) +
    scale_fill_gradient(low = "black", high = "white") +
    theme_void() +
    theme(legend.position = "none")
  
  saveRDS(p, here("cached_data/fig1_attention_map.rds"))
  
}


## ----false-check, fig.pos = "!h", fig.cap = 'A summary of the residual plot assessment evaluated on 200 null plots and 200 bootstrapped plots. (A) The true residual plot exhibiting a "left-triangle" shape. (B) The attention map highlights the top-right and bottom-right corners of the residual plot as the most influential.  (C) The density plot shows estimated distances for null (yellow) and bootstrapped (green) plots. The fitted model is not rejected because $\\hat{D} < Q_{null}(0.95)$. (D) Null and bootstrapped plots cluster together in the space defined by the first two principal components of the global pooling layer. The cluster also covers the true residual plot.', fig.height = 8, out.width = "80%"----
p1 <- my_vi$plot_resid(theme = theme_light()) + ggtitle(glue("(A) Residual plot")) 
p2 <- readRDS(here("cached_data/fig1_attention_map.rds")) + ggtitle("(B) Attention map")
p3 <- my_vi$summary_plot() +
  annotate("text", x = 2.8, y = 1.25, label = glue("p-value = {format(my_vi$p_value(), digits = 3)}")) +
  ggtitle("(C) Density plot of estimated D", subtitle = glue("Conventional p-value = {format(HETER_MODEL$test(dat)$p_value, digits = 3)}")) +
  theme(legend.position = "bottom") +
  labs(linetype = "") +
  xlab(expression(hat(D))) +
  scale_linetype(labels = c(expression(Q[null](0.95)), expression(observed~hat(D)))) +
  scale_fill_manual(values = c("#40B0A6", "#E1BE6A")) +
  scale_color_manual(values = c("#40B0A6", "#E1BE6A"))
pca_result <- my_vi$feature_pca()
pc1_per <- (attributes(pca_result)$sdev^2/256)[1]
pc2_per <- (attributes(pca_result)$sdev^2/256)[2]
p4 <- pca_result %>%
  mutate(set = ifelse(set == "boot", "Boot", set)) %>%
  mutate(set = ifelse(set == "null", "Null", set)) %>%
  ggplot() +
  geom_point(data = ~filter(.x, set != "observed"), aes(PC1, PC2, col = set), size = 2, alpha = 0.3) +
  geom_point(data = ~filter(.x, set == "observed"), aes(PC1, PC2), size = 4, col = "black", fill = "black") +
  geom_text(data = ~filter(.x, set == "observed"), aes(PC1, PC2, label = "Observed"), nudge_x = 10) +
  theme_light() +
  scale_color_brewer(palette = "Dark2") +
  xlab(glue("PC1 ({scales::percent(pc1_per)})")) +
  ylab(glue("PC2 ({scales::percent(pc2_per)})")) +
  labs(col = "") +
  theme(legend.position = "bottom") +
  ggtitle("(D) PCA for extracted features") +
  scale_color_manual(values = c("#40B0A6", "#E1BE6A"))

patchwork::wrap_plots(p1, p2, p3, p4, nrow = 2, widths = c(1, 1), heights = c(1, 1))


## ----false-lineup, fig.pos = "!h", fig.cap = 'A lineup of residual plots displaying "left-triangle" visual patterns. The true residual plot occupies position 10, yet there are no discernible visual patterns that distinguish it from the other plots.', fig.height = 8, out.width = "70%"----
pos <- 10
result <- tibble()
for (i in c(1:20)[-pos]) {
  result <- bind_rows(result,
                      my_vi$rotate_resid() %>%
                        mutate(k = i))
}
result <- bind_rows(my_vi$get_fitted_and_resid() %>% 
                      mutate(k = pos),
                    result)
l1 <- result %>%
  VI_MODEL$plot_lineup(remove_grid_line = TRUE,
                       remove_axis = TRUE,
                       theme = theme_bw(),
                       alpha = 0.3,
                       stroke = 0.3) +
  ggtitle("(A) Lineup for left-triangle")


## -------------------------------------------
housing <- read_csv(here("data/housing.csv"))
mod <- lm(MEDV ~ ., data = housing)

my_vi <- autovi::auto_vi(fitted_model = mod)


## -------------------------------------------
if (!file.exists(here("cached_data/boston_check.rds"))) {
  keras_model <- autovi::get_keras_model("vss_phn_32")
  my_vi$check(null_draws = 200L, boot_draws = 200L, keras_model = keras_model, extract_feature_from_layer = "global_max_pooling2d")
  saveRDS(my_vi$check_result, here("cached_data/boston_check.rds"))
}



## -------------------------------------------
my_vi$check_result <- readRDS(here("cached_data/boston_check.rds"))


## -------------------------------------------
if (!file.exists(here("cached_data/boston_attention_map.rds"))) {
  reticulate::py_set_seed(10086, disable_hash_randomization = TRUE)
  define_gradient()
  keras_mod <- autovi::get_keras_model("vss_phn_32")
  plot_path <- my_vi$plot_resid() %>%
    autovi::save_plot()
  input_auxiliary <- my_vi$auxiliary()

  reticulate::py$get_smooth_gradient(keras_mod = keras_mod, 
                                     plot_path = plot_path, 
                                     target_size = c(32L, 32L), 
                                     input_auxiliary = unlist(input_auxiliary),
                                     noise_level = 0.5,
                                     n = 1000L) -> smooth_g
  
  x_range <- ggplot_build(my_vi$plot_resid())$layout$panel_params[[1]]$x.range
  y_range <- ggplot_build(my_vi$plot_resid())$layout$panel_params[[1]]$y.range
  
  smooth_att_map <- smooth_g[[1]] |> 
    as.data.frame() |>
    mutate(row = rev(1:32)) |>
    pivot_longer(V1:V32, names_to = "column", values_to = "gradient") |>
    mutate(column = as.integer(gsub("V", "", column))) |>
    mutate(column = x_range[1] + column * (x_range[2] - x_range[1])/(32 - 1)) |>
    mutate(row = y_range[1] + row * (y_range[2] - y_range[1])/(32 - 1))
  
  scale_zero_one <- function(x) (x - min(x))/(max(x) - min(x))
  
  p <- ggplot() +
    geom_raster(data = mutate(smooth_att_map, gradient = scale_zero_one(gradient)), aes(column, row, fill = gradient), alpha = 0.7) +
    scale_fill_gradient(low = "black", high = "white") +
    theme_void() +
    theme(legend.position = "none")
  
  saveRDS(p, here("cached_data/boston_attention_map.rds"))
  
}


## ----boston-check, fig.pos = "!h", fig.cap = 'A summary of the residual plot assessment for the Boston housing fitted model evaluated on 200 null plots and 200 bootstrapped plots. (A) The true residual plot exhibiting a "U" shape. (B) The attention map highlights the central region of the "U" shape as the most influential part of the residual plot. (C) The density plot shows estimated distances for null (yellow) and bootstrapped (green) plots. The fitted model is rejected because $\\hat{D} \\geq Q_{null}(0.95)$. (D) Bootstrapped and null plots form distinct clusters in the space of the first two principal components from the global pooling layer, highlighting their visual separability.', fig.height = 8, out.width = "80%"----
p1 <- my_vi$plot_resid(theme = theme_light()) + ggtitle(glue("(A) Residual plot")) 
p2 <- readRDS(here("cached_data/boston_attention_map.rds")) + ggtitle("(B) Attention map")
p3 <- my_vi$summary_plot() +
    annotate("text", x = 6, y = 2.25, label = glue("p-value = {format(my_vi$p_value(), digits = 3)}")) +
  ggtitle("(C) Density plot of estimated D", subtitle = glue("Conventional p-value = {format(lmtest::resettest(mod)$p.value, digits = 3)}")) +
  theme(legend.position = "bottom") +
  labs(linetype = "") +
  xlab(expression(hat(D))) +
  scale_linetype(labels = c(expression(Q[null](0.95)), expression(observed~hat(D)))) +
  scale_fill_manual(values = c("#40B0A6", "#E1BE6A")) +
  scale_color_manual(values = c("#40B0A6", "#E1BE6A"))
pca_result <- my_vi$feature_pca()
pc1_per <- (attributes(pca_result)$sdev^2/256)[1]
pc2_per <- (attributes(pca_result)$sdev^2/256)[2]
p4 <- pca_result %>%
  mutate(set = ifelse(set == "boot", "Boot", set)) %>%
  mutate(set = ifelse(set == "null", "Null", set)) %>%
  ggplot() +
  geom_point(data = ~filter(.x, set != "observed"), aes(PC1, PC2, col = set), size = 2, alpha = 0.3) +
  geom_point(data = ~filter(.x, set == "observed"), aes(PC1, PC2), size = 4, col = "black", fill = "black") +
  geom_text(data = ~filter(.x, set == "observed"), aes(PC1, PC2, label = "Observed"), nudge_x = 7) +
  theme_light() +
  scale_color_brewer(palette = "Dark2") +
  xlab(glue("PC1 ({scales::percent(pc1_per)})")) +
  ylab(glue("PC2 ({scales::percent(pc2_per)})")) +
  labs(col = "") +
  theme(legend.position = "bottom") +
  ggtitle("(D) PCA for extracted features") +
  scale_color_manual(values = c("#40B0A6", "#E1BE6A"))

patchwork::wrap_plots(p1, p2, p3, p4, nrow = 2, widths = c(1, 1), heights = c(1, 1))


## ----boston-lineup, fig.pos = "!h", fig.cap = 'A lineup of residual plots for the Boston housing fitted model. The true residual plot is at position 7. It can be easily identified as the most different plot.', fig.height = 8, out.width = "70%"----
pos <- 7
result <- tibble()
for (i in c(1:20)[-pos]) {
  result <- bind_rows(result,
                      my_vi$rotate_resid() %>%
                        mutate(k = i))
}
result <- bind_rows(my_vi$get_fitted_and_resid() %>% 
                      mutate(k = pos),
                    result)
l2 <- result %>%
  VI_MODEL$plot_lineup(remove_grid_line = TRUE,
                       remove_axis = TRUE,
                       theme = theme_bw(),
                       alpha = 0.3,
                       stroke = 0.3) +
  ggtitle("(B) Lineup for Boston housing")


## -------------------------------------------
dino <- datasauRus::datasaurus_dozen %>% filter(dataset == "dino")
mod <- lm(y ~ ., data = select(dino, -dataset))

my_vi <- autovi::auto_vi(fitted_model = mod)


## -------------------------------------------
if (!file.exists(here("cached_data/dino_check.rds"))) {
  keras_model <- autovi::get_keras_model("vss_phn_32")
  my_vi$check(null_draws = 200L, boot_draws = 200L, keras_model = keras_model, extract_feature_from_layer = "global_max_pooling2d")
  saveRDS(my_vi$check_result, here("cached_data/dino_check.rds"))
}


## -------------------------------------------
my_vi$check_result <- readRDS(here("cached_data/dino_check.rds"))


## -------------------------------------------
if (!file.exists(here("cached_data/dino_attention_map.rds"))) {
  reticulate::py_set_seed(10086, disable_hash_randomization = TRUE)
  define_gradient()
  keras_mod <- autovi::get_keras_model("vss_phn_32")
  plot_path <- my_vi$plot_resid() %>%
    autovi::save_plot()
  input_auxiliary <- my_vi$auxiliary()

  reticulate::py$get_smooth_gradient(keras_mod = keras_mod, 
                                     plot_path = plot_path, 
                                     target_size = c(32L, 32L), 
                                     input_auxiliary = unlist(input_auxiliary),
                                     noise_level = 0.5,
                                     n = 1000L) -> smooth_g
  
  x_range <- ggplot_build(my_vi$plot_resid())$layout$panel_params[[1]]$x.range
  y_range <- ggplot_build(my_vi$plot_resid())$layout$panel_params[[1]]$y.range
  
  smooth_att_map <- smooth_g[[1]] |> 
    as.data.frame() |>
    mutate(row = rev(1:32)) |>
    pivot_longer(V1:V32, names_to = "column", values_to = "gradient") |>
    mutate(column = as.integer(gsub("V", "", column))) |>
    mutate(column = x_range[1] + column * (x_range[2] - x_range[1])/(32 - 1)) |>
    mutate(row = y_range[1] + row * (y_range[2] - y_range[1])/(32 - 1))
  
  scale_zero_one <- function(x) (x - min(x))/(max(x) - min(x))
  
  p <- ggplot() +
    geom_raster(data = mutate(smooth_att_map, gradient = scale_zero_one(gradient)), aes(column, row, fill = gradient), alpha = 0.7) +
    scale_fill_gradient(low = "black", high = "white") +
    theme_void() +
    theme(legend.position = "none")
  
  saveRDS(p, here("cached_data/dino_attention_map.rds"))
  
}


## ----dino-check, fig.pos = "!h", fig.cap = 'A summary of the residual plot assessment for the datasauRus fitted model evaluated on 200 null plots and 200 bootstrapped plots. (A) The residual plot exhibits a "dinosaur" shape. (B) The attention map highlights the dinosaur shape, indicating the model\'s decision is driven by human-recognizable features.(C) The density plot shows estimated distances for null (yellow) and bootstrapped (green) plots. The fitted model is rejected because $\\hat{D} \\geq Q_{null}(0.95)$. (D) The bootstrapped plots cluster at the corner of the null plot cluster, yet remain isolated in the space defined by the first two principal components of the global pooling layer.', fig.height = 8, out.width = "80%"----

p1 <- my_vi$plot_resid(theme = theme_light()) + ggtitle(glue("(A) Residual plot")) 
p2 <- readRDS(here("cached_data/dino_attention_map.rds")) + ggtitle("(B) Attention map")
p3 <- my_vi$summary_plot() +
    annotate("text", x = 6, y = 1.25, label = glue("p-value = {format(my_vi$p_value(), digits = 3)}")) +
  ggtitle("(C) Density plot of estimated D", subtitle = glue("RESET test p-value = {format(lmtest::resettest(mod)$p.value, digits = 3)}\n Breusch-Pagan test p-value = {format(lmtest::bptest(mod)$p.value, digits = 3)} \n Shapiro-Wilk test p-value = {format(shapiro.test(mod$residuals)$p.value, digits = 3)}")) +
  theme(legend.position = "bottom") +
  labs(linetype = "") +
  xlab(expression(hat(D))) +
  scale_linetype(labels = c(expression(Q[null](0.95)), expression(observed~hat(D)))) +
  scale_fill_manual(values = c("#40B0A6", "#E1BE6A")) +
  scale_color_manual(values = c("#40B0A6", "#E1BE6A"))
pca_result <- my_vi$feature_pca()
pc1_per <- (attributes(pca_result)$sdev^2/256)[1]
pc2_per <- (attributes(pca_result)$sdev^2/256)[2]
p4 <- pca_result %>%
  mutate(set = ifelse(set == "boot", "Boot", set)) %>%
  mutate(set = ifelse(set == "null", "Null", set)) %>%
  ggplot() +
  geom_point(data = ~filter(.x, set != "observed"), aes(PC1, PC2, col = set), size = 2, alpha = 0.3) +
  geom_point(data = ~filter(.x, set == "observed"), aes(PC1, PC2), size = 4, col = "black", fill = "black") +
  geom_text(data = ~filter(.x, set == "observed"), aes(PC1, PC2, label = "Observed"), nudge_x = -10) +
  theme_light() +
  scale_color_brewer(palette = "Dark2") +
  xlab(glue("PC1 ({scales::percent(pc1_per)})")) +
  ylab(glue("PC2 ({scales::percent(pc2_per)})")) +
  labs(col = "", shape = "") +
  theme(legend.position = "bottom", legend.box="vertical") +
  ggtitle("(D) PCA for extracted features") +
  scale_color_manual(values = c("#40B0A6", "#E1BE6A"))

patchwork::wrap_plots(p1, p2, p3, p4, nrow = 2, widths = c(1, 1), heights = c(1, 1))


## ----dino-lineup, fig.pos = "!h", fig.cap = 'A lineup of residual plots for the fitted model on the "dinosaur" dataset. The true residual plot is at position 17. It can be easily identified as the most different plot as the visual pattern is extremely artificial.', fig.height = 8, out.width = "70%"----
pos <- 17
result <- tibble()
for (i in c(1:20)[-pos]) {
  result <- bind_rows(result,
                      my_vi$rotate_resid() %>%
                        mutate(k = i))
}
result <- bind_rows(my_vi$get_fitted_and_resid() %>% 
                      mutate(k = pos),
                    result)
l3 <- result %>%
  VI_MODEL$plot_lineup(remove_grid_line = TRUE,
                       remove_axis = TRUE,
                       theme = theme_bw(),
                       alpha = 0.3,
                       stroke = 0.3) +
  ggtitle("(C) Lineup for datasauRus")


## ----lineup, fig.pos = "!h", fig.cap = 'Lineups of residual plots for the "left-triangle", "Boston housing", and "datasauRus" datasets. (A) True plot at position 10; no clear visual difference. (B) True plot at position 7; "U"-shape clearly stands out. (C) True plot at position 17; distinctly artificial and easily identifiable.', fig.width = 10, fig.height = 4----
patchwork::wrap_plots(l1, l2, l3, ncol = 3, widths = c(1, 1, 1))

