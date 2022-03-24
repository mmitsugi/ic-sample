FROM registry.access.redhat.com/ubi8/python-36

# Add application sources to a directory that the assemble script expects them
# and set permissions so that the container runs without root access
USER 0

## ppc64le need hdf5-devel for tensorflow
RUN yum -y install https://dl.fedoraproject.org/pub/epel/epel-release-latest-8.noarch.rpm && \
    yum -y install hdf5-devel && \
    yum -y clean all --enablerepo='*'

ADD . /tmp/src

RUN  chown -R 1001:0 /tmp/src && \
     chmod -R +x /tmp/src/.s2i/bin/assemble && \
     /usr/bin/fix-permissions /tmp/src
USER 1001

# Install the dependencies
RUN /tmp/src/.s2i/bin/assemble

# Set the default command for the resulting image
CMD /usr/libexec/s2i/run
