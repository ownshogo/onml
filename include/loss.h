#ifndef SRC_LOSS_H_
#define SRC_LOSS_H_

class loss {
public:
	virtual float compute(const float y, const float yhat) const = 0;
	virtual float gradient(const float y, const float yhat) const = 0;
	virtual ~loss() {};
};

class squared_loss : public loss {
public:
	float compute(const float y, const float yhat) const override;
	float gradient(const float y, const float yhat) const override;
};

class huber_loss : public loss {
public:
	huber_loss(const float delta);
	float compute(const float y, const float yhat) const override;
	float gradient(const float y, const float yhat) const override;
private:
	const float delta;
};

#endif /* SRC_LOSS_H_ */
