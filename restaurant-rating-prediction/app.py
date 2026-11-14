
import gradio as gr

from predict import predict_rating


def get_prediction(
    restaurant_name,
    cuisines,
    votes,
    average_cost,
    price_range,
    city,
    country_code,
    currency,
    table_booking,
    online_delivery,
    longitude,
    latitude,
):
    """Validate inputs and predict restaurant rating."""

    if not restaurant_name.strip():
        raise gr.Error("Enter the restaurant name.")

    if not cuisines.strip():
        raise gr.Error("Enter the restaurant cuisines.")

    if not city.strip():
        raise gr.Error("Enter the restaurant city.")

    if votes < 0 or average_cost < 0:
        raise gr.Error("Votes and average cost cannot be negative.")

    if not 1 <= price_range <= 4:
        raise gr.Error("Price range must be between 1 and 4.")

    if not 1 <= country_code:
        raise gr.Error("Enter a valid country code.")

    return predict_rating(
        restaurant_name=restaurant_name,
        cuisines=cuisines,
        votes=int(votes),
        average_cost=average_cost,
        price_range=int(price_range),
        city=city,
        country_code=int(country_code),
        currency=currency,
        table_booking=table_booking,
        online_delivery=online_delivery,
        longitude=longitude,
        latitude=latitude,
    )


with gr.Blocks(title="Restaurant Rating Prediction") as demo:

    gr.Markdown(
        """
        # Restaurant Rating Prediction System

        Predict restaurant ratings using a machine learning model
        trained on restaurant data.
        """
    )

    gr.Markdown(
        "Enter the restaurant information below to generate a prediction."
    )

    with gr.Row():

        with gr.Column():

            gr.Markdown("### Restaurant Information")

            restaurant_name = gr.Textbox(
                label="Restaurant Name",
                placeholder="e.g. Le Petit Souffle",
            )

            cuisines = gr.Textbox(
                label="Cuisines",
                placeholder="e.g. French, Japanese, Desserts",
            )

            city = gr.Textbox(
                label="City",
                placeholder="e.g. Makati City",
            )

            country_code = gr.Number(
                label="Country Code",
                value=162,
                precision=0,
            )

            currency = gr.Textbox(
                label="Currency",
                value="Botswana Pula(P)",
            )

        with gr.Column():

            gr.Markdown("### Restaurant Metrics")

            votes = gr.Number(
                label="Number of Votes",
                value=314,
                precision=0,
            )

            average_cost = gr.Number(
                label="Average Cost for Two",
                value=1100,
                precision=0,
            )

            price_range = gr.Slider(
                minimum=1,
                maximum=4,
                step=1,
                value=3,
                label="Price Range",
            )

            table_booking = gr.Radio(
                choices=["Yes", "No"],
                value="Yes",
                label="Has Table Booking?",
            )

            online_delivery = gr.Radio(
                choices=["Yes", "No"],
                value="No",
                label="Has Online Delivery?",
            )

    gr.Markdown("### Geographic Information")

    with gr.Row():

        longitude = gr.Number(
            label="Longitude",
            value=121.027535,
        )

        latitude = gr.Number(
            label="Latitude",
            value=14.565443,
        )

    predict_button = gr.Button(
        "Predict Restaurant Rating",
        variant="primary",
    )

    prediction_output = gr.Number(
        label="Predicted Restaurant Rating (out of 5)",
        precision=2,
    )

    predict_button.click(
        fn=get_prediction,
        inputs=[
            restaurant_name,
            cuisines,
            votes,
            average_cost,
            price_range,
            city,
            country_code,
            currency,
            table_booking,
            online_delivery,
            longitude,
            latitude,
        ],
        outputs=prediction_output,
    )

    gr.Markdown(
        """
        ---
        **Powered by Machine Learning | Random Forest Regressor**
        """
    )


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
    )
