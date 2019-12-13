#ifndef SRC_LOSS_H_
#define SRC_LOSS_H_

#include <algorithm>
#include <cmath>

namespace onml {
class loss
{
public:
  virtual float compute(float y, float yhat) const = 0;
  virtual float gradient(float y, float yhat) const = 0;
  virtual ~loss(){};
};

class squared_loss : public loss
{
public:
  float compute(float y, float yhat) const override;
  float gradient(float y, float yhat) const override;
};

class huber_loss : public loss
{
public:
  huber_loss(float delta);
  float compute(float y, float yhat) const override;
  float gradient(float y, float yhat) const override;

private:
  const float delta;
};

class logistic_loss : public loss
{
public:
  float compute(float y, float yhat) const override;
  float gradient(float y, float yhat) const override;
};

class hinge_loss : public loss
{
public:
  float compute(float y, float yhat) const override;
  float gradient(float y, float yhat) const override;
};
}
#endif /* SRC_LOSS_H_ */
