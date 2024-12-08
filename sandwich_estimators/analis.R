library('knitr')                        #создание отчетов
opts_chunk$set(fig.align = 'center')    #выравнивание картинок по центру
library('ggplot2')                      #графики
library('sandwich')                     #оценка Var для гетероскедастичности
library('lmtest')                       #тест Бройша-Пагана
library('dplyr')                        #манипуляции с данными

filename <- 'flats_moscow.txt'
flats <- subset(read.table(filename, header=T), , -n)

ggplot(flats)+
geom_point(aes(x=totsp, y=price))+
labs(x = 'Full flat square, sq.m', y='Flat price, $1000', title='Flat price in Moscow')

# Регрессия стоимости на общую площадь 
m1 <- lm(price ~ totsp, flats)
summary(m1)

# Доверительные интервалы для коэффициентов из регрессии
confint(m1)

#Бетта с крышой
vcov(m1)

#Если не нужно всё summary, а только бетта с крышей
coeftest(m1)

#Графическое отображение гетероскедастичности
m1.st.resid <- rstandard(m1)
ggplot(aes(x=totsp, y=abs(m1.st.resid)), data=flats)+
    geom_point(alpha=0.2)+
    labs(x='Full flat square, sq.m', y=expression(paste('Standart resids, ', s[i])),
    title='Graphical geteroscedastics')

#Тест Бройша-Пагана (версия Коэнкера)
bptest(m1)
#Тест Бройша-Пагана (версия классика)
bptest(m1, studentize=F)

#тест Уайта (White test), частный случай современной модификации теста Бройша-Пагана:
bptest(m1, varformula = ~totsp + I(totsp^2), data=flats)