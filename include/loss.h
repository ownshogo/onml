#ifndef SRC_LOSS_H_
#define SRC_LOSS_H_

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
}
#endif /* SRC_LOSS_H_ */
