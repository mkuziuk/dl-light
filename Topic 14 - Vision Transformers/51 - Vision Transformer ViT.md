# Vision Transformer ViT

Source: `DL_exam.pdf`, Question 51

Original question:

> Vision Transformer. Patch embedding, positional encoding, self-attention для изображений, сравнение ViT и CNN.

## Главная идея

`Vision Transformer` рассматривает изображение как последовательность токенов: картинку режут на патчи, проецируют их в embeddings, добавляют позицию и пропускают через `Transformer encoder`. Выигрыш - глобальный `self-attention`; цена - слабые image priors и квадратичная стоимость по числу патчей.

## Минимум для ответа

- Вход $x \in \mathbb{R}^{H \times W \times C}$ делится на патчи $P \times P$; $N = HW/P^2$.
- `Patch embedding`: flatten патча $P^2C$ и линейная проекция в $D$. `Conv2d(kernel_size=P, stride=P)` дает тот же эффект для неперекрывающихся патчей.
- Для классификации добавляют обучаемый `[CLS]` token; его выход идет в classifier head.
- `Positional embeddings` обязательны: без них attention не знает 2D-порядок.
- Блок ViT: `LN -> MHA -> residual -> LN -> MLP/FFN -> residual` в `pre-LN` форме.
- CNN имеет локальность, sharing weights и трансляционную эквивариантность; ViT слабее в priors, лучше масштабируется на больших данных, но хуже data-efficient с нуля.

## Формулы / схема

$$
Z_0 = [z_{\mathrm{cls}}; x_p^1E; \ldots; x_p^NE] + E_{\mathrm{pos}},
\quad E \in \mathbb{R}^{P^2C \times D}
$$

$$
\mathrm{Attention}(Q,K,V)=\mathrm{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V
$$

$$
Z'_\ell = Z_\ell + \mathrm{MHA}(\mathrm{LN}(Z_\ell)), \quad
Z_{\ell+1}=Z'_\ell+\mathrm{MLP}(\mathrm{LN}(Z'_\ell))
$$

Pipeline: image -> patches -> projection -> `[CLS]` + positions -> $L$ encoder blocks -> `[CLS]` -> classifier -> logits.

## Диаграмма

```mermaid
flowchart LR
    A["Image H x W x C"] --> B["P x P patches"]
    B --> C["Flatten + Linear projection"]
    C --> D["Patch tokens"]
    D --> E["Add CLS token"]
    E --> F["Add positional embeddings"]
    F --> G["Transformer encoder blocks"]
    G --> H["CLS output"]
    H --> I["Classifier head"]
    I --> J["Class logits"]
```

## Уточнения экзаменатора

- Зачем positional encoding? Чтобы сохранить порядок патчей; self-attention без позиций перестановочно-эквивариантен.
- Почему attention дорогой? Матрица внимания имеет размер примерно $(N+1)^2$, поэтому уменьшение $P$ резко увеличивает стоимость.
- Чем attention отличается от свертки? Attention связывает любые пары патчей по содержимому; свертка применяет локальное ядро.
- Где еще применяют ViT? Как backbone для detection, segmentation, self-supervised и vision-language моделей.

## Частые ошибки

- Думать, что Transformer сам знает 2D-положение.
- Путать patch tokens с отдельными пикселями: в базовом ViT attention идет по патчам.
- Называть ViT всегда лучше CNN: без pretraining CNN часто устойчивее.
- Забывать `[CLS]` token в классической схеме.
- Считать attention maps строгим причинным объяснением решения.
