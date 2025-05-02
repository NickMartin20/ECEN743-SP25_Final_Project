import evaluate

bleu = evaluate.load("bleu")

predictions = ["the cat is on the mat", "there is a cat on the mat"]
references = [["the cat is on the mat"], ["a cat is on the mat"]]

score = bleu.compute(predictions=predictions, references=references)
print(f"BLEU: {score['bleu']:.4f}")
