library(ggplot2)

train <- read.csv('./error_20_10_train_clean.csv', header=T)
val <- read.csv('./error_20_10_val_clean.csv', header=T)

epoch <- train$epoch
train_mse <- train$MSE
val_mse <- val$MSE

results <- data.frame(epoch, train_mse,val_mse)
ggplot(results) + geom_line(aes(x=epoch,y=train_mse,color="Entrenamiento")) + geom_line(aes(x=epoch,y=val_mse,color="Validación")) + scale_color_discrete('Fase') + xlab('Epoch') + ylab('MSE') + theme(legend.key.size = unit(0.5, 'cm'), legend.title = element_text(size=10))