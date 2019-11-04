Detecting Child Sexual Abuse Material
*************************************

Using `perception` and a subscription to Thorn's Safer service,
you can easily check for child sexual abuse material against a database of known bad content
**without** having to send any images to a third party. You do this by sending compact, irreversible
image hashes to get matches with a high degree of precision. We support matching using
16x16 PHash hashes and md5 hashes.

See usage example below. Please contact info@getsafer.io to discuss Thorn's Safer service
and subscription options and visit `getsafer.io <https://getsafer.io/>`_ to learn more.

.. code-block:: python

    from perception import tools
    matcher = tools.SaferMatcher(
        api_key='YOUR_API_KEY',
        url='MATCHING_SERVICE_URL'
    )
    matches = matcher.match(['myfile.jpg'])

In some cases, you may have a username/password instead of an API key, in which case
you can pass those instead (see API documentation for details).