#include "loss.h"

namespace onml {
float
squared_loss::compute(float y, float yhat) const
{
  return (y - yhat) * (y - yhat);
}

float
squared_loss::gradient(float y, float yhat) const
{
  return 2 * (yhat - y);
}

huber_loss::huber_loss(float delta)
  : delta(delta)
{}

float
huber_loss::compute(float y, float yhat) const
{
  float diff = std::abs(y - yhat);
  if (diff <= this->delta) {
    return diff * diff / 2;
  }
  return this->delta * (diff - this->delta / 2);
}

float
huber_loss::gradient(float y, float yhat) const
{
  float diff = std::abs(y - yhat);
  if (diff <= this->delta) {
    return yhat - y;
  }
  return -this->delta;
}

float
logistic_loss::compute(float y, float yhat) const
{
  if (y == 1.) {
    return -std::log(yhat);
  }
  return -std::log(1 - yhat);
}

float
logistic_loss::gradient(float y, float yhat) const
{
  return yhat - y;
}
}
