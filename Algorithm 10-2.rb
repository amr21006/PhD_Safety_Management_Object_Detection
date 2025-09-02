Initialize generators G_AB and G_BA and discriminators D_A and D_B
for number of training iterations do
  # Update discriminators D_A and D_B
  for k steps do
    Sample minibatch of m images {x1, x2, ..., xm} from data distribution X_A
    Sample minibatch of m images {y1, y2, ..., ym} from data distribution X_B
    Update D_A by ascending its stochastic gradient:
      ∇ [1/m Σ max(0, 1-D_A(x)) + max(0, 1+D_A(G_AB(y)))]
    Update D_B by ascending its stochastic gradient:
      ∇ [1/m Σ max(0, 1-D_B(y)) + max(0, 1+D_B(G_BA(x)))]
  end for

  # Update generators G_AB and G_BA
  Sample minibatch of m noise samples {z1, z2, ..., zm} from noise prior p(z)
  Update G_AB and G_BA by descending their stochastic gradient:
    ∇ [1/m Σ max(0, 1-D_A(G_AB(y))) + max(0, 1-D_B(G_BA(x))) + λ||x-G_BA(G_AB(y))||1 + λ||y-G_AB(G_BA(x))||1]
end for

