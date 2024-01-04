from flask import Flask, jsonify, request, redirect, url_for
from flasgger import Swagger
from marketing_compliance import MarketingCompliance

app = Flask(__name__)
swagger = Swagger(app)
compliance_entity = MarketingCompliance()


@app.route('/server_liveness', methods=['GET'])
def server_liveness():
    """
    This method checks if the server is live
    :return:
    """
    return jsonify({'status': 'Server is live'}), 200


@app.route('/swagger', methods=['GET'])
def swagger_ui():
    """
    Serve Swagger UI for API documentation.

    ---
    responses:
      302:
        description: Redirect to Swagger UI
    """
    return redirect(url_for('flasgger.apidocs'))


@app.route('/api/v1/get_report', methods=['POST'])
def get_results():
    """
    This API gets the url as input and check the content in the page against a compliance policy.

    ---
    parameters:
      - name: url
        in: formData
        type: string
        required: true
        description: The URL for which results are requested
    responses:
      200:
        description: JSON response with findings
      400:
        description: Bad request if URL is not provided
    """
    data = request.form
    url = data.get('url')

    if not url:
        return jsonify({'error': 'URL parameter is required'}), 400

    results = compliance_entity.get_compliance_report(url)

    return jsonify(results), 200


if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5000)
